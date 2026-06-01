"""Lens-specific image and prompt encoding."""

from __future__ import annotations

from typing import List, Tuple

import torch
from einops import rearrange
from PIL import Image
from torchvision import transforms


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def vae_bn_params(vae) -> Tuple[torch.Tensor, torch.Tensor]:
    bn = vae.bn
    mean = bn.running_mean.view(1, -1, 1, 1)
    var = bn.running_var.view(1, -1, 1, 1)
    std = torch.sqrt(var + vae.config.batch_norm_eps)
    shift = -mean
    scale = 1.0 / std
    return shift, scale


@torch.no_grad()
def encode_images_to_latents(
    pipe,
    images: List[Image.Image],
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode PIL images to Lens sequence latents [B, seq, 128]."""
    transform = transforms.Compose(
        [
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    batch = torch.stack([transform(img.convert("RGB")) for img in images]).to(device=device, dtype=dtype)

    vae = pipe.vae
    if vae.device != device:
        vae.to(device)

    encoded = vae.encode(batch)
    if hasattr(encoded, "latent_dist"):
        vae_latents = encoded.latent_dist.sample()
    else:
        vae_latents = encoded

    shift, scale = vae_bn_params(vae)
    shift = shift.to(device=device, dtype=dtype)
    scale = scale.to(device=device, dtype=dtype)

    latent_h = height // pipe.vae_scale_factor
    latent_w = width // pipe.vae_scale_factor

    # Inverse of LensPipeline._decode (rearrange -> patchify -> normalize -> unpatchify).
    x_norm = patchify_latents(vae_latents)
    x = (x_norm + shift) * scale
    rearranged = unpatchify_latents(x)
    seq = rearrange(
        rearranged,
        "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
        p1=2,
        p2=2,
        h=latent_h,
        w=latent_w,
    )
    return seq.contiguous()


@torch.no_grad()
def decode_latents_to_pil(
    pipe,
    latents: torch.Tensor,
    height: int,
    width: int,
) -> List[Image.Image]:
    """Decode Lens sequence latents to PIL images."""
    latent_h = height // pipe.vae_scale_factor
    latent_w = width // pipe.vae_scale_factor

    latents_4d = rearrange(
        latents,
        "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
        p1=2,
        p2=2,
        h=latent_h,
        w=latent_w,
    ).to(dtype=pipe.vae.dtype)

    shift, scale = vae_bn_params(pipe.vae)
    shift = shift.to(device=latents_4d.device, dtype=latents_4d.dtype)
    scale = scale.to(device=latents_4d.device, dtype=latents_4d.dtype)

    x = patchify_latents(latents_4d)
    x = x / scale - shift
    x = unpatchify_latents(x)
    decoded = pipe.vae.decode(x).sample
    decoded = decoded.clamp(-1.0, 1.0)
    decoded = (decoded + 1.0) * 127.5
    decoded = decoded.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return [Image.fromarray(im) for im in decoded]


@torch.no_grad()
def encode_prompt_features(
    pipe,
    prompts: List[str],
    device: torch.device,
    max_sequence_length: int = 512,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Return multi-layer GPT-OSS features and attention mask."""
    features, mask = pipe._get_text_embeddings(prompts, max_sequence_length, device)
    return features, mask
