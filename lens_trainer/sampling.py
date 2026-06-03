"""Mid-training sampling via the Lens pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import torch

from lens_trainer.config import SampleConfig, TrainConfig
from lens_trainer.console import sample as log_sample


@dataclass(frozen=True)
class ResolvedPrompt:
    name: str
    text: str


def _slugify(name: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug[:48] if slug else fallback


def resolve_prompts(raw: Sequence[Any], trigger_word: str) -> List[ResolvedPrompt]:
    """Parse YAML prompt entries (plain strings or {name, prompt} maps)."""
    resolved: List[ResolvedPrompt] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            name = f"{index:02d}"
            text = item
        elif isinstance(item, dict):
            text = item.get("prompt") or item.get("text")
            if not text:
                raise ValueError(f"Sample prompt at index {index} missing 'prompt'")
            name = str(item.get("name") or f"{index:02d}")
        else:
            raise ValueError(
                f"Sample prompt at index {index} must be a string or mapping, got {type(item)}"
            )
        rendered = text.replace("[trigger]", trigger_word)
        resolved.append(ResolvedPrompt(name=_slugify(name, f"prompt_{index:02d}"), text=rendered))
    return resolved


def should_sample_at_step(step: int, cfg: TrainConfig, *, skip_step_zero: bool = True) -> bool:
    if skip_step_zero and step == 0:
        return False
    if step <= cfg.sample_early_until and cfg.sample_every_early > 0:
        if step % cfg.sample_every_early == 0:
            return True
    if cfg.sample_every > 0 and step % cfg.sample_every == 0:
        return True
    return False


def release_inference(pipe) -> None:
    """Drop sampling hooks and move pipeline modules back to CPU."""
    from lens_trainer.trainer import _release_gpu_memory

    _release_gpu_memory(pipe)


def prepare_for_inference(
    pipe,
    transformer,
    device: torch.device,
    *,
    low_vram: bool,
) -> None:
    """Configure the pipeline for preview generation.

    On 16GB GPUs (``low_vram=True``) the text encoder stays on CPU and only the
    DiT + VAE are offloaded to CUDA one at a time. Callers must pass pre-encoded
    ``prompt_embeds`` into ``pipe()`` — see ``_encode_prompt_for_low_vram``.
    """
    pipe.transformer = transformer
    transformer.eval()

    if hasattr(pipe, "remove_all_hooks"):
        pipe.remove_all_hooks()

    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    transformer.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if low_vram and device.type == "cuda":
        from accelerate import cpu_offload_with_hook

        torch_device = device
        pipe._offload_device = torch_device
        pipe._offload_gpu_id = torch_device.index or 0
        hook = None
        _, hook = cpu_offload_with_hook(transformer, torch_device, prev_module_hook=None)
        _, hook = cpu_offload_with_hook(pipe.vae, torch_device, prev_module_hook=hook)
        pipe._all_hooks = [hook] if hook is not None else []
        pipe.transformer = transformer
    elif device.type == "cuda":
        pipe.to(device)
        pipe.transformer = transformer
    else:
        pipe.transformer = transformer


@torch.no_grad()
def _encode_prompt_for_low_vram(
    pipe,
    prompt: str,
    max_sequence_length: int,
    device: torch.device,
    *,
    disable_mxfp4: bool,
) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Encode one prompt for low-VRAM sampling (DiT/VAE off GPU during TE work).

    MXFP4 Triton kernels require CUDA — cannot use ``device=cpu`` when MXFP4 is active.
    """
    if hasattr(pipe, "remove_all_hooks"):
        pipe.remove_all_hooks()

    te_hook = None
    if disable_mxfp4 or device.type != "cuda":
        encode_device = torch.device("cpu")
        pipe.text_encoder.to("cpu")
    else:
        from lens_trainer.trainer import (
            _attach_text_encoder_offload,
            _detach_text_encoder_offload,
            _park_pipeline_for_text_cache,
        )

        _park_pipeline_for_text_cache(pipe)
        encode_device = device
        te_hook = _attach_text_encoder_offload(pipe.text_encoder, device)

    try:
        prompt_embeds, prompt_mask = pipe._get_text_embeddings(
            [prompt], max_sequence_length, encode_device
        )
    finally:
        if te_hook is not None:
            from lens_trainer.trainer import _detach_text_encoder_offload

            _detach_text_encoder_offload(pipe.text_encoder, te_hook)
        pipe.text_encoder.to("cpu")

    negative_prompt_embeds = [feat.new_zeros(feat.shape) for feat in prompt_embeds]
    negative_prompt_mask = torch.zeros_like(prompt_mask, dtype=torch.bool)
    return prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask


def _inference_device(pipe, fallback: torch.device) -> torch.device:
    if hasattr(pipe, "_offload_device"):
        return pipe._offload_device
    if hasattr(pipe, "_execution_device"):
        return pipe._execution_device
    return fallback


@torch.no_grad()
def run_sample_set(
    pipe,
    cfg: SampleConfig,
    output_dir: Path,
    step: int,
    prompts: Sequence[Any],
    tag: str,
    *,
    device: torch.device,
    low_vram: bool = False,
    disable_mxfp4: bool = True,
) -> None:
    """Generate one PNG per prompt and save with stable names."""
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_prompts(prompts, cfg.trigger_word)
    if not resolved:
        return

    log_sample(
        f"Sampling {len(resolved)} prompt(s) at step {step} ({tag}) -> {output_dir}"
    )
    transformer = pipe.transformer

    for offset, item in enumerate(resolved):
        if low_vram:
            release_inference(pipe)
            (
                prompt_embeds,
                prompt_mask,
                negative_prompt_embeds,
                negative_prompt_mask,
            ) = _encode_prompt_for_low_vram(
                pipe,
                item.text,
                cfg.max_sequence_length,
                device,
                disable_mxfp4=disable_mxfp4,
            )
            prepare_for_inference(pipe, transformer, device, low_vram=True)
            exec_device = _inference_device(pipe, device)
            prompt_embeds = [tensor.to(exec_device) for tensor in prompt_embeds]
            prompt_mask = prompt_mask.to(exec_device)
            negative_prompt_embeds = [
                tensor.to(exec_device) for tensor in negative_prompt_embeds
            ]
            negative_prompt_mask = negative_prompt_mask.to(exec_device)
            pipe_kwargs = {
                # Lens encode_prompt always parses `prompt` before checking embeds.
                "prompt": "",
                "prompt_embeds": prompt_embeds,
                "prompt_mask": prompt_mask,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt_mask": negative_prompt_mask,
            }
        else:
            prepare_for_inference(pipe, transformer, device, low_vram=False)
            exec_device = _inference_device(pipe, device)
            pipe_kwargs = {"prompt": item.text}

        seed = cfg.seed + offset if cfg.walk_seed else cfg.seed
        generator = torch.Generator(device=exec_device).manual_seed(seed)
        try:
            result = pipe(
                height=cfg.height,
                width=cfg.width,
                num_inference_steps=cfg.steps,
                guidance_scale=cfg.cfg,
                generator=generator,
                max_sequence_length=cfg.max_sequence_length,
                **pipe_kwargs,
            )
        finally:
            if low_vram:
                release_inference(pipe)

        filename = f"step_{step:06d}_{tag}_{item.name}.png"
        out_path = output_dir / filename
        result.images[0].save(out_path)
        log_sample(f"  saved {out_path.name}")


@torch.no_grad()
def run_sampling(
    pipe,
    cfg: SampleConfig,
    output_dir: Path,
    step: int,
    *,
    device: torch.device,
    low_vram: bool = False,
    tag: str = "lora",
    prompts: Sequence[Any] | None = None,
    disable_mxfp4: bool = True,
) -> None:
    run_sample_set(
        pipe,
        cfg,
        output_dir,
        step,
        prompts if prompts is not None else cfg.prompts,
        tag=tag,
        device=device,
        low_vram=low_vram,
        disable_mxfp4=disable_mxfp4,
    )


@torch.no_grad()
def run_baseline_control_samples(
    pipe,
    base_transformer,
    cfg: SampleConfig,
    output_dir: Path,
    device: torch.device,
    cpu_offload: bool,
    disable_mxfp4: bool = True,
) -> None:
    """Sample the full prompt list with the untouched base DiT (no LoRA) at step 0."""
    if not cfg.baseline_control or not cfg.prompts:
        return

    release_inference(pipe)
    pipe.transformer = base_transformer
    try:
        run_sample_set(
            pipe,
            cfg,
            output_dir,
            step=0,
            prompts=cfg.prompts,
            tag="control",
            device=device,
            low_vram=cpu_offload,
            disable_mxfp4=disable_mxfp4,
        )
    finally:
        release_inference(pipe)
