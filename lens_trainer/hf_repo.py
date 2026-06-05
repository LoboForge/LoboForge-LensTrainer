"""Assemble a Hugging Face–layout Lens repo from local weights + Hub downloads."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from safetensors import safe_open

TRANSFORMER_MARKERS = ("img_in.weight", "proj_out.weight", "transformer_blocks.0.")
VAE_MARKERS = ("bn.running_mean", "encoder.conv_in.weight", "decoder.conv_out.weight")
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
# Real Lens-Base shards are hundreds of MB; tiny files are git-lfs pointers or HTML stubs.
MIN_WEIGHT_BYTES = 1_000_000


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


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size > 4096:
        return False
    with path.open("rb") as handle:
        return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def is_loadable_safetensors_shard(path: Path) -> bool:
    """True when path is a real safetensors weight file (not LFS pointer / HTML stub)."""
    if not path.is_file():
        return False
    if path.stat().st_size < MIN_WEIGHT_BYTES:
        return False
    if is_git_lfs_pointer(path):
        return False
    try:
        with safe_open(str(path), framework="pt") as handle:
            next(iter(handle.keys()), None)
        return True
    except Exception:
        return False


def _shard_paths(component_dir: Path, index_name: str, single_name: str) -> list[Path]:
    single = component_dir / single_name
    if single.is_file():
        return [single]
    index_path = component_dir / index_name
    if not index_path.is_file():
        return []
    with index_path.open("r", encoding="utf-8") as handle:
        index_data = json.load(handle)
    weight_map = index_data.get("weight_map") or {}
    if not weight_map:
        return []
    return [component_dir / name for name in sorted(set(weight_map.values()))]


def diagnose_hf_repo(path: Path) -> list[str]:
    """Return human-readable problems with a Lens HF folder (empty if loadable)."""
    path = path.expanduser()
    problems: list[str] = []

    if not (path / "model_index.json").is_file():
        problems.append(f"missing {path / 'model_index.json'}")
        return problems

    for label, component, index_name, single_name in (
        ("text_encoder", path / "text_encoder", "model.safetensors.index.json", "model.safetensors"),
        (
            "transformer",
            path / "transformer",
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model.safetensors",
        ),
    ):
        shards = _shard_paths(component, index_name, single_name)
        if not shards:
            problems.append(f"{label}: no weight shards under {component}")
            continue
        for shard in shards:
            if not shard.is_file():
                problems.append(f"{label}: missing {shard.name}")
            elif is_git_lfs_pointer(shard):
                problems.append(
                    f"{label}: {shard.name} is a git-lfs pointer ({shard.stat().st_size} bytes) — "
                    "re-download with huggingface_hub, not git clone without git-lfs"
                )
            elif shard.stat().st_size < MIN_WEIGHT_BYTES:
                problems.append(
                    f"{label}: {shard.name} too small ({shard.stat().st_size} bytes) — truncated or wrong file"
                )
            elif not is_loadable_safetensors_shard(shard):
                problems.append(f"{label}: {shard.name} is not a valid safetensors checkpoint")

    vae = path / "vae" / "diffusion_pytorch_model.safetensors"
    if not vae.is_file():
        problems.append(f"vae: missing {vae}")
    elif is_git_lfs_pointer(vae):
        problems.append(f"vae: git-lfs pointer ({vae.stat().st_size} bytes)")
    elif not is_loadable_safetensors_shard(vae):
        problems.append(f"vae: invalid or truncated {vae.name}")

    return problems


def _component_weights_present(component_dir: Path, index_name: str, single_name: str) -> bool:
    shards = _shard_paths(component_dir, index_name, single_name)
    return bool(shards) and all(is_loadable_safetensors_shard(shard) for shard in shards)


def is_complete_hf_repo(path: Path) -> bool:
    """Return True if path looks like a loadable Lens HF pipeline folder."""
    path = path.expanduser()
    if not (path / "model_index.json").is_file():
        return False
    if not _component_weights_present(path / "text_encoder", "model.safetensors.index.json", "model.safetensors"):
        return False
    if not _component_weights_present(
        path / "transformer",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors",
    ):
        return False
    vae = path / "vae" / "diffusion_pytorch_model.safetensors"
    return is_loadable_safetensors_shard(vae)


def resolve_model_repo(repo_id: str) -> str:
    """If repo_id is a relative/absolute path to a complete HF folder, use it as-is."""
    candidate = Path(repo_id).expanduser()
    if candidate.is_dir() and is_complete_hf_repo(candidate):
        return str(candidate.resolve())
    return repo_id


def repo_needs_redownload(path: Path) -> bool:
    """True when the folder is missing, incomplete, or contains git-lfs pointer stubs."""
    path = path.expanduser()
    if not path.is_dir():
        return True
    if is_complete_hf_repo(path):
        return False
    return bool(diagnose_hf_repo(path)) or (path / ".git").is_dir()


def wipe_lens_repo_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _hf_cli_download(repo_id: str, output_dir: Path) -> None:
    hf = shutil.which("hf")
    if not hf:
        venv_hf = Path(os.environ.get("VIRTUAL_ENV", "")) / "bin" / "hf"
        if venv_hf.is_file():
            hf = str(venv_hf)
    if not hf:
        huggingface_cli = shutil.which("huggingface-cli")
        if huggingface_cli:
            subprocess.run(
                [huggingface_cli, "download", repo_id, "--local-dir", str(output_dir)],
                check=True,
            )
            return
        raise RuntimeError("hf / huggingface-cli not found in PATH")
    subprocess.run([hf, "download", repo_id, "--local-dir", str(output_dir)], check=True)


def download_lens_base_from_hub(
    output_dir: Path,
    repo_id: str = "microsoft/Lens-Base",
    *,
    force: bool = False,
) -> Path:
    """
    Download a complete Lens-Base folder via Hugging Face Hub.

    Never use ``git clone`` on the Hub repo — without git-lfs you only get pointer
    files and training fails with ``HeaderTooLarge``.
    """
    output_dir = output_dir.expanduser().resolve()

    if is_complete_hf_repo(output_dir) and not force:
        print(f"Lens-Base already complete: {output_dir}")
        return output_dir

    if output_dir.exists():
        problems = diagnose_hf_repo(output_dir)
        if (output_dir / ".git").is_dir():
            problems.insert(
                0,
                "models/Lens-Base was git-cloned — remove it and use Hub download instead",
            )
        if problems:
            print(f"Removing broken Lens-Base at {output_dir}:")
            for line in problems:
                print(f"  - {line}")
        wipe_lens_repo_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} → {output_dir} (multi-GB weight files; not git-lfs pointers) ...")
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            force_download=True,
        )
    except Exception as exc:
        print(f"snapshot_download failed ({exc}); trying hf download ...")
        wipe_lens_repo_dir(output_dir)
        _hf_cli_download(repo_id, output_dir)

    if not is_complete_hf_repo(output_dir):
        problems = diagnose_hf_repo(output_dir)
        detail = "\n".join(f"  - {p}" for p in problems) or "  - unknown"
        raise RuntimeError(
            f"Lens-Base at {output_dir} is still not loadable after Hub download.\n{detail}\n"
            "Check: HF_TOKEN or `hf auth login`, and license accepted at "
            f"https://huggingface.co/{repo_id}"
        )

    manifest = {
        "repo_id": repo_id,
        "output_dir": str(output_dir),
        "source": "huggingface_hub",
    }
    with (output_dir / ".lens_trainer_assembled.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Ready: {output_dir}")
    return output_dir


def assemble_lens_repo(
    output_dir: Path,
    repo_id: str = "microsoft/Lens-Base",
    transformer_source: Path | None = None,
    vae_source: Path | None = None,
    use_symlink: bool = True,
    force: bool = False,
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

    if transformer_source is None and vae_source is None:
        return download_lens_base_from_hub(output_dir, repo_id=repo_id, force=force)

    if force or repo_needs_redownload(output_dir):
        wipe_lens_repo_dir(output_dir)
    else:
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
        local_dir_use_symlinks=False,
        force_download=bool(force),
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
