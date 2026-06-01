"""Mid-training sampling via the Lens pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch

from lens_trainer.config import SampleConfig


def apply_trigger(prompts: List[str], trigger_word: str) -> List[str]:
    return [p.replace("[trigger]", trigger_word) for p in prompts]


@torch.no_grad()
def run_sampling(
    pipe,
    cfg: SampleConfig,
    output_dir: Path,
    step: int,
    lora_model=None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = apply_trigger(cfg.prompts, cfg.trigger_word)
    generator = torch.Generator(device=pipe._execution_device).manual_seed(cfg.seed)

    transformer = pipe.transformer
    if lora_model is not None:
        pipe.transformer = lora_model

    try:
        result = pipe(
            prompt=prompts,
            height=cfg.height,
            width=cfg.width,
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.cfg,
            generator=generator,
        )
        for idx, image in enumerate(result.images):
            image.save(output_dir / f"step_{step:06d}_{idx:02d}.png")
    finally:
        pipe.transformer = transformer
