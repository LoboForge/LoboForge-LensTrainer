"""Main Lens LoRA training loop."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lens_trainer.config import TrainerConfig, resolve_output_dir
from lens_trainer.dataset import LensDataset
from lens_trainer.encoding import (
    encode_images_to_latents,
    encode_prompt_features,
    torch_dtype,
)
from lens_trainer.lora import attach_lora, save_lora
from lens_trainer.sampling import run_sampling
from lens_trainer.scheduler import flow_match_noisy_latents, flow_match_target, sample_timesteps


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model, cfg: TrainerConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.train.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(
                params,
                lr=cfg.train.learning_rate,
                weight_decay=cfg.train.weight_decay,
            )
        except ImportError:
            pass
    return torch.optim.AdamW(
        params,
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )


def load_lens_pipeline(cfg: TrainerConfig):
    from lens import LensGptOssEncoder, LensPipeline

    dtype = torch_dtype(cfg.model.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_encoder_kwargs = {"subfolder": "text_encoder", "dtype": dtype}
    try:
        from transformers import Mxfp4Config

        text_encoder_kwargs["quantization_config"] = Mxfp4Config(
            dequantize=cfg.model.disable_mxfp4
        )
    except ImportError:
        pass

    text_encoder = LensGptOssEncoder.from_pretrained(
        cfg.model.repo_id,
        **text_encoder_kwargs,
    )
    pipe = LensPipeline.from_pretrained(
        cfg.model.repo_id,
        text_encoder=text_encoder,
        torch_dtype=dtype,
    )

    if cfg.model.cpu_offload and device.type == "cuda":
        pipe.enable_model_cpu_offload()
    elif device.type == "cuda":
        pipe.to(device)

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.eval()
    pipe.text_encoder.eval()

    return pipe, device, dtype


@torch.no_grad()
def precompute_caches(
    pipe,
    dataset: LensDataset,
    device: torch.device,
    dtype: torch.dtype,
    cache_text: bool,
) -> None:
    for index in tqdm(range(len(dataset)), desc="Precomputing caches"):
        resolution = dataset.resolution

        if dataset.cfg.cache_latents and dataset.load_latent_cache(index) is None:
            image = dataset.get_image(index)
            latents = encode_images_to_latents(
                pipe,
                [image],
                height=resolution,
                width=resolution,
                device=pipe._execution_device,
                dtype=dtype,
            )
            dataset.save_latent_cache(index, latents[0])

        if cache_text and dataset.load_text_cache(index) is None:
            caption = dataset.get_caption(index)
            features, mask = encode_prompt_features(
                pipe,
                [caption],
                device=pipe._execution_device,
                max_sequence_length=dataset.cfg.max_sequence_length,
            )
            dataset.save_text_cache(index, features, mask[0:1])


def collate_batch(batch: List[dict]) -> dict:
    return {
        "index": [item["index"] for item in batch],
        "caption": [item["caption"] for item in batch],
        "resolution": batch[0]["resolution"],
    }


class LensTrainer:
    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self.output_dir = resolve_output_dir(cfg)
        self.cache_dir = self.output_dir / "cache"
        self.samples_dir = self.output_dir / "samples"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "config.resolved.json").write_text(
            json.dumps(
                {
                    "job": cfg.job.__dict__,
                    "model": cfg.model.__dict__,
                    "dataset": cfg.dataset.__dict__,
                    "lora": cfg.lora.__dict__,
                    "train": cfg.train.__dict__,
                    "sample": cfg.sample.__dict__,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def run(self) -> None:
        cfg = self.cfg
        set_seed(cfg.train.seed)
        pipe, device, dtype = load_lens_pipeline(cfg)

        dataset = LensDataset(cfg.dataset, cache_dir=self.cache_dir)
        precompute_caches(
            pipe,
            dataset,
            device,
            dtype,
            cache_text=cfg.model.cache_text_embeddings,
        )

        if cfg.train.gradient_checkpointing:
            pipe.transformer.enable_gradient_checkpointing()

        lora_model = attach_lora(
            pipe.transformer,
            rank=cfg.lora.rank,
            alpha=cfg.lora.alpha,
            dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
        )
        lora_model.train()

        optimizer = build_optimizer(lora_model, cfg)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            drop_last=len(dataset) > cfg.train.batch_size,
        )

        num_train_timesteps = pipe.scheduler.config.num_train_timesteps
        global_step = 0
        epoch = 0
        progress = tqdm(total=cfg.train.steps, desc="Training")

        while global_step < cfg.train.steps:
            epoch += 1
            for batch in dataloader:
                if global_step >= cfg.train.steps:
                    break

                resolution = batch["resolution"]
                latent_h = resolution // pipe.vae_scale_factor
                latent_w = resolution // pipe.vae_scale_factor

                latents_list = []
                text_features_list: List[List[torch.Tensor]] = []
                masks = []

                for batch_idx, index in enumerate(batch["index"]):
                    cached_latent = dataset.load_latent_cache(index)
                    if cached_latent is not None:
                        latents_list.append(cached_latent.unsqueeze(0))
                    else:
                        image = dataset.get_image(index)
                        encoded = encode_images_to_latents(
                            pipe,
                            [image],
                            height=resolution,
                            width=resolution,
                            device=pipe._execution_device,
                            dtype=dtype,
                        )
                        latents_list.append(encoded)

                    if cfg.model.cache_text_embeddings:
                        cached_text = dataset.load_text_cache(index)
                        assert cached_text is not None
                        text_features_list.append(
                            [f.unsqueeze(0) for f in cached_text["features"]]
                        )
                        masks.append(cached_text["mask"].unsqueeze(0))
                    else:
                        features, mask = encode_prompt_features(
                            pipe,
                            [batch["caption"][batch_idx]],
                            device=pipe._execution_device,
                            max_sequence_length=cfg.dataset.max_sequence_length,
                        )
                        text_features_list.append([f.unsqueeze(0) for f in features])
                        masks.append(mask.unsqueeze(0))

                latents = torch.cat(latents_list, dim=0).to(
                    device=pipe._execution_device, dtype=dtype
                )
                mask = torch.cat(masks, dim=0).to(device=pipe._execution_device)

                num_layers = len(text_features_list[0])
                encoder_hidden_states = [
                    torch.cat([sample[layer_idx] for sample in text_features_list], dim=0).to(
                        device=pipe._execution_device, dtype=dtype
                    )
                    for layer_idx in range(num_layers)
                ]

                noise = torch.randn_like(latents)
                timesteps = sample_timesteps(
                    latents.shape[0],
                    num_train_timesteps,
                    device=pipe._execution_device,
                    timestep_type=cfg.train.timestep_type,
                )
                noisy_latents = flow_match_noisy_latents(
                    latents, noise, timesteps, num_train_timesteps
                )
                target = flow_match_target(noise, latents)

                pred = lora_model(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=mask,
                    timestep=timesteps.to(dtype=dtype) / 1000.0,
                    img_shapes=[(1, latent_h, latent_w)],
                )

                loss = F.mse_loss(pred.float(), target.float())
                scaled_loss = loss / cfg.train.gradient_accumulation_steps
                scaled_loss.backward()

                if (global_step + 1) % cfg.train.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        lora_model.parameters(), cfg.train.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                progress.update(1)
                progress.set_postfix(loss=f"{loss.item():.4f}", epoch=epoch)
                global_step += 1

                if global_step % cfg.train.save_every == 0 or global_step == cfg.train.steps:
                    ckpt_path = self.checkpoints_dir / f"lora_step_{global_step:06d}.safetensors"
                    save_lora(
                        lora_model,
                        ckpt_path,
                        metadata={
                            "base_model": cfg.model.repo_id,
                            "step": global_step,
                            "rank": cfg.lora.rank,
                            "alpha": cfg.lora.alpha,
                        },
                    )

                if global_step % cfg.train.sample_every == 0:
                    run_sampling(
                        pipe,
                        cfg.sample,
                        self.samples_dir,
                        step=global_step,
                        lora_model=lora_model,
                    )

        progress.close()
        final_path = self.output_dir / "lora_final.safetensors"
        save_lora(
            lora_model,
            final_path,
            metadata={
                "base_model": cfg.model.repo_id,
                "step": global_step,
                "rank": cfg.lora.rank,
                "alpha": cfg.lora.alpha,
            },
        )
