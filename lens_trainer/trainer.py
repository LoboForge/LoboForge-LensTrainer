"""Main Lens LoRA training loop."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lens_trainer.config import TrainerConfig, resolve_output_dir
from lens_trainer.branding import print_startup_banner
from lens_trainer.console import (
    checkpoint as log_checkpoint,
    error as log_error,
    resume as log_resume,
    tqdm_bar,
    warn as log_warn,
)
from lens_trainer.hf_repo import resolve_model_repo
from lens_trainer.dataset import LensDataset
from lens_trainer.encoding import (
    encode_images_to_latents,
    encode_prompt_features,
    torch_dtype,
)
from lens_trainer.lens_patches import (
    enable_lens_gradient_checkpointing,
    prepare_lens_transformer_for_load,
)
from lens_trainer.lora import (
    attach_lora,
    find_resume_checkpoint,
    load_lora_weights,
    resolve_resume_checkpoint,
    save_lora,
)
from lens_trainer.sampling import (
    run_baseline_control_samples,
    run_sampling,
    should_sample_at_step,
)
from lens_trainer.scheduler import flow_match_noisy_latents, flow_match_target, sample_timesteps

_LOSS_FLUSH_EVERY = 25


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


def _teardown_cpu_offload(pipe) -> None:
    if hasattr(pipe, "remove_all_hooks"):
        pipe.remove_all_hooks()


def _release_gpu_memory(pipe) -> None:
    """Move all pipeline modules to CPU and drop cached CUDA allocations."""
    import gc

    _teardown_cpu_offload(pipe)
    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    pipe.transformer.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _attach_text_encoder_offload(text_encoder, device: torch.device):
    from accelerate import cpu_offload_with_hook

    text_encoder.to("cpu")
    _, hook = cpu_offload_with_hook(text_encoder, device)
    return hook


def _detach_text_encoder_offload(text_encoder, hook) -> None:
    if hook is not None:
        hook.offload()
    if hasattr(text_encoder, "_hf_hook"):
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(text_encoder, recurse=True)
    text_encoder.to("cpu")


def prepare_for_training(pipe, lora_model, device: torch.device) -> None:
    """Park TE/VAE on CPU and keep the LoRA-wrapped transformer on the train device."""
    _teardown_cpu_offload(pipe)
    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    lora_model.to(device)
    lora_model.train()
    pipe.transformer = lora_model


def prepare_for_sampling(pipe, lora_model, device: torch.device, cpu_offload: bool) -> None:
    """Configure the pipeline for mid-training preview generation."""
    from lens_trainer.sampling import prepare_for_inference

    prepare_for_inference(pipe, lora_model, device, low_vram=cpu_offload)


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

    model_repo = resolve_model_repo(cfg.model.repo_id)
    text_encoder = LensGptOssEncoder.from_pretrained(
        model_repo,
        **text_encoder_kwargs,
    )
    pipe = LensPipeline.from_pretrained(
        model_repo,
        text_encoder=text_encoder,
        torch_dtype=dtype,
    )
    prepare_lens_transformer_for_load(pipe.transformer)

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
    cache_text: bool,
    device: torch.device,
    dtype: torch.dtype,
    cpu_offload: bool = False,
    disable_mxfp4: bool = True,
) -> None:
    """Precompute latent and/or text caches on disk.

    CPU offload hooks from ``load_lens_pipeline`` conflict with manually moving
    the VAE to CUDA (can abort with cublasLt errors). Latent encoding tears
    those hooks down; text encoding uses TE-only layer offload or CPU.
    """
    needs_latents = any(
        dataset.cfg.cache_latents and dataset.load_latent_cache(index) is None
        for index in range(len(dataset))
    )
    needs_text = cache_text and any(
        dataset.load_text_cache(index) is None for index in range(len(dataset))
    )

    if needs_latents:
        _release_gpu_memory(pipe)
        for index in tqdm_bar(
            "cache_latent", range(len(dataset)), desc="Precomputing latents"
        ):
            if dataset.cfg.cache_latents and dataset.load_latent_cache(index) is None:
                try:
                    train_h, train_w = dataset.get_training_size(index)
                    encoded = _encode_batch_latents(
                        pipe,
                        dataset,
                        index,
                        train_h,
                        train_w,
                        device,
                        dtype,
                    )
                    dataset.save_latent_cache(index, encoded[0])
                except (OSError, RuntimeError, ValueError) as exc:
                    name = dataset.items[index].image_path.name
                    log_warn(f"Skipping latent cache for {name}: {exc}")
        _release_gpu_memory(pipe)

    if needs_text:
        _release_gpu_memory(pipe)

        # Full-pipeline cpu_offload keeps the DiT (~13GB) in the hook chain and
        # tries to swap whole components. For cache building we only need the TE.
        # disable_mxfp4=true dequantizes GPT-OSS to bf16 (~40GB RAM); it never
        # fits whole on 16GB VRAM, so default to CPU unless the user has MXFP4.
        encode_on_cpu = disable_mxfp4 or device.type != "cuda"
        te_hook = None
        if encode_on_cpu:
            encode_device = torch.device("cpu")
        else:
            encode_device = device
            if cpu_offload:
                te_hook = _attach_text_encoder_offload(pipe.text_encoder, device)

        try:
            for index in tqdm_bar(
                "cache_text", range(len(dataset)), desc="Precomputing text"
            ):
                if dataset.load_text_cache(index) is None:
                    try:
                        caption = dataset.get_caption(index)
                        features, mask = encode_prompt_features(
                            pipe,
                            [caption],
                            device=encode_device,
                            max_sequence_length=dataset.cfg.max_sequence_length,
                        )
                        dataset.save_text_cache(index, features, mask[0:1].bool())
                    except (OSError, RuntimeError, ValueError) as exc:
                        name = dataset.items[index].image_path.name
                        log_warn(f"Skipping text cache for {name}: {exc}")
        finally:
            if te_hook is not None:
                _detach_text_encoder_offload(pipe.text_encoder, te_hook)
            _release_gpu_memory(pipe)

    _release_gpu_memory(pipe)


def collate_batch(batch: List[dict]) -> dict:
    heights = [item["height"] for item in batch]
    widths = [item["width"] for item in batch]
    if len(set(zip(heights, widths))) > 1:
        raise ValueError(
            "Batch contains mixed training sizes. Use train.batch_size: 1 or a "
            "dataset where every image shares the same snapped height×width."
        )
    return {
        "index": [item["index"] for item in batch],
        "caption": [item["caption"] for item in batch],
        "height": heights[0],
        "width": widths[0],
    }


def _apply_sample_size_from_dataset(cfg: TrainerConfig, dataset: LensDataset) -> None:
    """When sample height/width are 0, match the dataset's dominant training size."""
    sample_h, sample_w = dataset.primary_sample_size
    if cfg.sample.height <= 0:
        cfg.sample.height = sample_h
    if cfg.sample.width <= 0:
        cfg.sample.width = sample_w


@torch.no_grad()
def _encode_batch_latents(
    pipe,
    dataset: LensDataset,
    index: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    image = dataset.get_image(index)
    pipe.vae.to(device)
    encoded = encode_images_to_latents(
        pipe,
        [image],
        height=height,
        width=width,
        device=device,
        dtype=dtype,
    )
    pipe.vae.to("cpu")
    return encoded


@torch.no_grad()
def _encode_batch_text(
    pipe,
    caption: str,
    device: torch.device,
    max_sequence_length: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    pipe.text_encoder.to(device)
    features, mask = encode_prompt_features(
        pipe,
        [caption],
        device=device,
        max_sequence_length=max_sequence_length,
    )
    pipe.text_encoder.to("cpu")
    return [f.unsqueeze(0) for f in features], mask.unsqueeze(0).bool()


class LensTrainer:
    def __init__(self, cfg: TrainerConfig, config_path: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.output_dir = resolve_output_dir(cfg)
        self.cache_dir = self.output_dir / "cache"
        self.samples_dir = self.output_dir / "samples"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._write_resolved_config()

    def _write_resolved_config(self) -> None:
        payload = {
            "job": self.cfg.job.__dict__,
            "model": self.cfg.model.__dict__,
            "dataset": self.cfg.dataset.__dict__,
            "lora": self.cfg.lora.__dict__,
            "train": self.cfg.train.__dict__,
            "sample": self.cfg.sample.__dict__,
        }
        (self.output_dir / "config.resolved.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        if self.config_path is not None and self.config_path.exists():
            shutil.copy2(self.config_path, self.output_dir / "config.yaml")

    def _checkpoint_metadata(self, global_step: int) -> dict[str, object]:
        cfg = self.cfg
        return {
            "base_model": cfg.model.repo_id,
            "step": global_step,
            "rank": cfg.lora.rank,
            "alpha": cfg.lora.alpha,
        }

    def _flush_loss_log(self, loss_log: list[dict]) -> None:
        loss_path = self.output_dir / "loss.json"
        loss_path.write_text(json.dumps(loss_log, indent=2), encoding="utf-8")

    def _persist_checkpoint(
        self,
        lora_model,
        global_step: int,
        *,
        path: Path,
        update_latest: bool = True,
    ) -> None:
        save_lora(lora_model, path, metadata=self._checkpoint_metadata(global_step))
        if update_latest:
            latest = self.output_dir / "lora_latest.safetensors"
            shutil.copy2(path, latest)

    def _save_step_checkpoint(self, lora_model, global_step: int) -> Path:
        ckpt_path = self.checkpoints_dir / f"lora_step_{global_step:06d}.safetensors"
        self._persist_checkpoint(lora_model, global_step, path=ckpt_path)
        return ckpt_path

    def _save_emergency_checkpoint(self, lora_model, global_step: int) -> None:
        if global_step <= 0:
            return
        path = self.output_dir / "lora_emergency.safetensors"
        self._persist_checkpoint(lora_model, global_step, path=path)
        log_checkpoint(f"Emergency checkpoint saved: {path} (step {global_step})")

    def _resolve_resume(
        self, resume_from: str, lora_model
    ) -> tuple[int, bool]:
        """Load checkpoint if present. Returns (global_step, did_resume)."""
        token = resume_from.strip()
        if not token:
            return 0, False

        if token.lower() in {"latest", "auto"}:
            ckpt = find_resume_checkpoint(
                output_dir=self.output_dir,
                checkpoints_dir=self.checkpoints_dir,
            )
            if ckpt is None:
                log_warn(
                    "train.resume_from is set to "
                    f"'{token}' but no checkpoint exists under "
                    f"{self.checkpoints_dir} or lora_latest / lora_emergency — "
                    "starting training from step 0."
                )
                return 0, False
        else:
            ckpt = resolve_resume_checkpoint(
                token,
                output_dir=self.output_dir,
                checkpoints_dir=self.checkpoints_dir,
            )

        resume_info = load_lora_weights(lora_model, ckpt)
        global_step = int(resume_info["step"])
        log_resume(
            f"Resumed from {ckpt.name} at step {global_step} "
            f"({resume_info['loaded_keys']} LoRA tensors)"
        )
        ckpt_rank = resume_info.get("rank")
        ckpt_alpha = resume_info.get("alpha")
        cfg = self.cfg
        if ckpt_rank is not None and str(ckpt_rank) != str(cfg.lora.rank):
            log_warn(
                f"checkpoint rank={ckpt_rank} differs from config rank={cfg.lora.rank}"
            )
        if ckpt_alpha is not None and str(ckpt_alpha) != str(cfg.lora.alpha):
            log_warn(
                f"checkpoint alpha={ckpt_alpha} differs from config alpha={cfg.lora.alpha}"
            )
        if global_step >= cfg.train.steps:
            raise SystemExit(
                f"Checkpoint is already at step {global_step}, "
                f"but train.steps={cfg.train.steps}. Increase train.steps to continue."
            )
        return global_step, True

    def run(self) -> None:
        cfg = self.cfg
        print_startup_banner(cfg, output_dir=str(self.output_dir))
        dataset_path = Path(cfg.dataset.folder_path)
        if not dataset_path.is_dir():
            raise SystemExit(
                f"Dataset folder not found: {dataset_path}. "
                "Set dataset.folder_path in your YAML or pass "
                "--set dataset.folder_path=/path/to/images"
            )

        set_seed(cfg.train.seed)
        pipe, device, dtype = load_lens_pipeline(cfg)

        dataset = LensDataset(cfg.dataset, cache_dir=self.cache_dir)
        _apply_sample_size_from_dataset(cfg, dataset)
        precompute_caches(
            pipe,
            dataset,
            cache_text=cfg.model.cache_text_embeddings,
            device=device,
            dtype=dtype,
            cpu_offload=cfg.model.cpu_offload,
            disable_mxfp4=cfg.model.disable_mxfp4,
        )

        base_transformer = pipe.transformer
        resume_path = (cfg.train.resume_from or "").strip()
        if resume_path and find_resume_checkpoint(
            output_dir=self.output_dir,
            checkpoints_dir=self.checkpoints_dir,
        ):
            log_resume("Checkpoint found — skipping step-0 control samples")
        elif cfg.sample.baseline_control:
            run_baseline_control_samples(
                pipe,
                base_transformer,
                cfg.sample,
                self.samples_dir,
                device,
                cfg.model.cpu_offload,
            )

        lora_model = attach_lora(
            pipe.transformer,
            rank=cfg.lora.rank,
            alpha=cfg.lora.alpha,
            dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
        )
        if cfg.train.gradient_checkpointing:
            enable_lens_gradient_checkpointing(lora_model)

        global_step = 0
        if resume_path:
            global_step, _ = self._resolve_resume(resume_path, lora_model)

        prepare_for_training(pipe, lora_model, device)

        optimizer = build_optimizer(lora_model, cfg)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            drop_last=cfg.train.batch_size > 1,
        )

        num_train_timesteps = pipe.scheduler.config.num_train_timesteps
        accum_step = 0
        epoch = 0
        loss_log: list[dict] = []
        loss_path = self.output_dir / "loss.json"
        if loss_path.exists():
            try:
                loss_log = json.loads(loss_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log_warn(
                    f"could not parse existing {loss_path.name}, starting fresh loss log"
                )

        progress = tqdm_bar(
            "train",
            total=cfg.train.steps,
            initial=global_step,
            desc="Training",
        )

        interrupted = False
        try:
            while global_step < cfg.train.steps:
                epoch += 1
                for batch in dataloader:
                    if global_step >= cfg.train.steps:
                        break

                    height = int(batch["height"])
                    width = int(batch["width"])
                    latent_h = height // pipe.vae_scale_factor
                    latent_w = width // pipe.vae_scale_factor

                    latents_list = []
                    text_features_list: List[List[torch.Tensor]] = []
                    masks = []

                    for batch_idx, index in enumerate(batch["index"]):
                        cached_latent = dataset.load_latent_cache(index)
                        if cached_latent is not None:
                            latents_list.append(cached_latent.unsqueeze(0))
                        else:
                            encoded = _encode_batch_latents(
                                pipe,
                                dataset,
                                index,
                                height,
                                width,
                                device,
                                dtype,
                            )
                            latents_list.append(encoded)
                            if cfg.dataset.cache_latents:
                                dataset.save_latent_cache(index, encoded[0])

                        if cfg.model.cache_text_embeddings:
                            cached_text = dataset.load_text_cache(index)
                            if cached_text is None:
                                raise RuntimeError(
                                    f"Missing text cache for sample {index}; "
                                    "re-run with cache_text_embeddings enabled."
                                )
                            text_features_list.append(cached_text["features"])
                            mask = cached_text["mask"].bool()
                            if mask.dim() == 3 and mask.shape[0] == 1:
                                mask = mask.squeeze(0)
                            masks.append(mask)
                        else:
                            features, mask = _encode_batch_text(
                                pipe,
                                batch["caption"][batch_idx],
                                device,
                                cfg.dataset.max_sequence_length,
                            )
                            text_features_list.append(features)
                            masks.append(mask)

                    latents = torch.cat(latents_list, dim=0).to(device=device, dtype=dtype)
                    mask = torch.cat(masks, dim=0).to(device=device, dtype=torch.bool)

                    num_layers = len(text_features_list[0])
                    encoder_hidden_states = [
                        torch.cat(
                            [sample[layer_idx] for sample in text_features_list],
                            dim=0,
                        ).to(device=device, dtype=dtype)
                        for layer_idx in range(num_layers)
                    ]

                    noise = torch.randn_like(latents)
                    timesteps = sample_timesteps(
                        latents.shape[0],
                        num_train_timesteps,
                        device=device,
                        timestep_type=cfg.train.timestep_type,
                    )
                    noisy_latents = flow_match_noisy_latents(
                        latents, noise, timesteps, num_train_timesteps
                    ).to(dtype=dtype)
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
                    accum_step += 1

                    if accum_step >= cfg.train.gradient_accumulation_steps:
                        torch.nn.utils.clip_grad_norm_(
                            lora_model.parameters(), cfg.train.max_grad_norm
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        accum_step = 0

                    progress.update(1)
                    progress.set_postfix(loss=f"{loss.item():.4f}", epoch=epoch)
                    loss_log.append({"step": global_step + 1, "loss": loss.item()})
                    global_step += 1

                    if global_step % _LOSS_FLUSH_EVERY == 0:
                        self._flush_loss_log(loss_log)

                    if global_step % cfg.train.save_every == 0 or global_step == cfg.train.steps:
                        ckpt_path = self._save_step_checkpoint(lora_model, global_step)
                        log_checkpoint(f"Checkpoint saved: {ckpt_path}")

                    if should_sample_at_step(global_step, cfg.train):
                        try:
                            run_sampling(
                                pipe,
                                cfg.sample,
                                self.samples_dir,
                                step=global_step,
                                tag="lora",
                                device=device,
                                low_vram=cfg.model.cpu_offload,
                            )
                        finally:
                            prepare_for_training(pipe, lora_model, device)

        except KeyboardInterrupt:
            interrupted = True
            log_error("Training interrupted (Ctrl+C).")

        if accum_step > 0:
            torch.nn.utils.clip_grad_norm_(
                lora_model.parameters(), cfg.train.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        progress.close()
        self._flush_loss_log(loss_log)

        if interrupted:
            self._save_emergency_checkpoint(lora_model, global_step)
            raise SystemExit(
                f"Stopped at step {global_step}. "
                f"Resume with --set train.resume_from=latest "
                f"(uses checkpoints/, lora_latest, or lora_emergency)."
            )

        final_path = self.output_dir / "lora_final.safetensors"
        self._persist_checkpoint(lora_model, global_step, path=final_path)
        log_checkpoint(f"Final LoRA saved: {final_path}")
