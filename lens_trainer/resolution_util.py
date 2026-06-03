"""Training resolution helpers (auto-detect, VAE alignment, optional downscale)."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Tuple

from PIL import Image, ImageOps


def read_image_size(path) -> tuple[int, int]:
    """Return (width, height) from file without loading full decode when possible."""
    with Image.open(path) as img:
        img.load()
        img = ImageOps.exif_transpose(img)
        return img.size


def snap_training_size(
    width: int,
    height: int,
    *,
    vae_scale_factor: int = 16,
    max_edge: int = 0,
) -> tuple[int, int]:
    """Return (height, width) for training, divisible by ``vae_scale_factor``.

    Optionally scales down so the long edge does not exceed ``max_edge`` (aspect preserved).
    """
    if width < 1 or height < 1:
        raise ValueError(f"invalid image size: {width}x{height}")

    if max_edge > 0:
        long_edge = max(width, height)
        if long_edge > max_edge:
            scale = max_edge / long_edge
            width = max(1, int(round(width * scale)))
            height = max(1, int(round(height * scale)))

    def snap(value: int) -> int:
        snapped = round(value / vae_scale_factor) * vae_scale_factor
        return max(vae_scale_factor, snapped)

    return snap(height), snap(width)


def uses_auto_resolution(resolution: int) -> bool:
    return int(resolution) <= 0


def summarize_sizes(
    sizes: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], Counter[tuple[int, int]]]:
    """Return (most common (height, width), full counter)."""
    counter: Counter[tuple[int, int]] = Counter(sizes)
    if not counter:
        raise ValueError("no image sizes to summarize")
    return counter.most_common(1)[0][0], counter
