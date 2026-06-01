"""Flow-match timestep sampling utilities."""

from __future__ import annotations

import math

import torch


def sample_timesteps(
    batch_size: int,
    num_train_timesteps: int,
    device: torch.device,
    timestep_type: str = "shift",
) -> torch.Tensor:
    """Sample training timesteps on 0..num_train_timesteps scale."""
    if timestep_type == "uniform":
        t = torch.randint(0, num_train_timesteps, (batch_size,), device=device)
        return t

    # Shifted logit-normal style sampling (ai-toolkit-ish default for flow models).
    u = torch.rand(batch_size, device=device)
    shifted = u ** 2
    t = (shifted * (num_train_timesteps - 1)).long().clamp(0, num_train_timesteps - 1)
    return t


def flow_match_noisy_latents(
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    num_train_timesteps: int,
) -> torch.Tensor:
    """Linear flow-matching interpolation."""
    sigma = (timesteps.float() / num_train_timesteps).view(-1, 1, 1)
    return (1.0 - sigma) * latents + sigma * noise


def flow_match_target(noise: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    return noise - latents
