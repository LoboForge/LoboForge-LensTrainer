"""Dataset with optional latent and text-embedding caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from lens_trainer.config import DatasetConfig
from lens_trainer.console import dataset as log_dataset
from lens_trainer.console import warn as log_warn
from lens_trainer.resolution_util import (
    read_image_size,
    snap_training_size,
    summarize_sizes,
    uses_auto_resolution,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VAE_SCALE_FACTOR = 16


@dataclass
class DatasetItem:
    image_path: Path
    caption: str
    source_width: int
    source_height: int
    train_height: int
    train_width: int


@dataclass(frozen=True)
class SkippedImage:
    path: Path
    reason: str


def validate_image_file(path: Path) -> tuple[int, int]:
    """Raise if the file is missing, empty, or not a readable RGB image.

    Returns (width, height).
    """
    if not path.is_file():
        raise FileNotFoundError("file not found")
    if path.stat().st_size == 0:
        raise OSError("empty file")

    with Image.open(path) as img:
        img.load()
        rgb = img.convert("RGB")
        width, height = rgb.size
        if width < 1 or height < 1:
            raise OSError(f"invalid dimensions: {width}x{height}")
        return width, height


def _training_size_for_image(
    width: int,
    height: int,
    *,
    auto: bool,
    square_resolution: int,
    max_training_edge: int,
) -> tuple[int, int]:
    if auto:
        return snap_training_size(
            width,
            height,
            vae_scale_factor=VAE_SCALE_FACTOR,
            max_edge=max_training_edge,
        )
    side = int(square_resolution)
    return side, side


def discover_items(
    folder: Path,
    caption_ext: str,
    *,
    auto_resolution: bool,
    square_resolution: int,
    max_training_edge: int,
) -> tuple[List[DatasetItem], List[SkippedImage]]:
    items: List[DatasetItem] = []
    skipped: List[SkippedImage] = []

    for image_path in sorted(folder.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        caption_path = image_path.with_suffix(f".{caption_ext}")
        if not caption_path.exists():
            continue
        caption = caption_path.read_text(encoding="utf-8").strip()
        if not caption:
            skipped.append(SkippedImage(image_path, "empty caption"))
            log_warn(f"Skipping {image_path.name}: empty caption")
            continue
        try:
            width, height = validate_image_file(image_path)
            train_height, train_width = _training_size_for_image(
                width,
                height,
                auto=auto_resolution,
                square_resolution=square_resolution,
                max_training_edge=max_training_edge,
            )
        except (OSError, UnidentifiedImageError, FileNotFoundError, ValueError) as exc:
            skipped.append(SkippedImage(image_path, str(exc)))
            log_warn(f"Skipping {image_path.name}: {exc}")
            continue
        items.append(
            DatasetItem(
                image_path=image_path,
                caption=caption,
                source_width=width,
                source_height=height,
                train_height=train_height,
                train_width=train_width,
            )
        )

    if not items:
        detail = (
            f"No valid image/caption pairs found under {folder}. "
            f"Expected image.jpg + image.{caption_ext} side by side."
        )
        if skipped:
            detail += f" Skipped {len(skipped)} invalid pair(s)."
        raise ValueError(detail)
    if skipped:
        log_dataset(f"Dataset ready: {len(items)} valid pair(s), {len(skipped)} skipped.")
    return items, skipped


class LensDataset(Dataset):
    def __init__(
        self,
        cfg: DatasetConfig,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.folder = Path(cfg.folder_path)
        self.auto_resolution = uses_auto_resolution(cfg.resolution)
        self.square_resolution = int(cfg.resolution) if not self.auto_resolution else 0
        self.max_training_edge = int(cfg.max_training_edge)
        self.items, self.skipped = discover_items(
            self.folder,
            cfg.caption_ext,
            auto_resolution=self.auto_resolution,
            square_resolution=self.square_resolution or 1024,
            max_training_edge=self.max_training_edge,
        )
        self.cache_dir = cache_dir
        self._log_resolution_summary()
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._write_manifest()

    def _log_resolution_summary(self) -> None:
        primary, counter = summarize_sizes(
            (item.train_height, item.train_width) for item in self.items
        )
        mode = "auto (per-image)" if self.auto_resolution else f"square {self.square_resolution}"
        h, w = primary
        bucket_count = len(counter)
        if bucket_count == 1:
            log_dataset(
                f"Training resolution {mode}: {len(self.items)} image(s), "
                f"all at {h}×{w} px (H×W)."
            )
        else:
            parts = ", ".join(
                f"{count}×{height}×{width}"
                for (height, width), count in counter.most_common()
            )
            log_dataset(
                f"Training resolution {mode}: {len(self.items)} image(s), "
                f"{bucket_count} size bucket(s) — {parts} (H×W). "
                "Use batch_size: 1 if buckets differ."
            )

    @property
    def primary_sample_size(self) -> tuple[int, int]:
        """Most common (height, width) in the dataset — used for mid-run previews."""
        size, _ = summarize_sizes(
            (item.train_height, item.train_width) for item in self.items
        )
        return size

    def get_training_size(self, index: int) -> tuple[int, int]:
        item = self.items[index]
        return item.train_height, item.train_width

    def _write_manifest(self) -> None:
        manifest = {
            "folder": str(self.folder),
            "resolution_mode": "auto" if self.auto_resolution else "square",
            "square_resolution": self.square_resolution,
            "max_training_edge": self.max_training_edge,
            "count": len(self.items),
            "skipped": [
                {"image": str(entry.path), "reason": entry.reason}
                for entry in self.skipped
            ],
            "items": [
                {
                    "image": str(it.image_path),
                    "caption": it.caption,
                    "source_width": it.source_width,
                    "source_height": it.source_height,
                    "train_height": it.train_height,
                    "train_width": it.train_width,
                }
                for it in self.items
            ],
        }
        (self.cache_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def __len__(self) -> int:
        return len(self.items)

    def _cache_key(self, item: DatasetItem, kind: str) -> str:
        payload = (
            f"{kind}|{item.image_path}|{item.caption}|"
            f"{item.train_height}x{item.train_width}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, item: DatasetItem, kind: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / kind / f"{self._cache_key(item, kind)}.pt"

    def get_image(self, index: int) -> Image.Image:
        item = self.items[index]
        try:
            with Image.open(item.image_path) as img:
                img.load()
                return img.convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to load validated image {item.image_path}: {exc}"
            ) from exc

    def get_caption(self, index: int) -> str:
        return self.items[index].caption

    def load_latent_cache(self, index: int) -> Optional[torch.Tensor]:
        if not self.cfg.cache_latents or self.cache_dir is None:
            return None
        path = self._cache_path(self.items[index], "latents")
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        return None

    def save_latent_cache(self, index: int, latents: torch.Tensor) -> None:
        if not self.cfg.cache_latents or self.cache_dir is None:
            return
        path = self._cache_path(self.items[index], "latents")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents.detach().cpu(), path)

    def load_text_cache(self, index: int) -> Optional[dict]:
        if self.cache_dir is None:
            return None
        path = self._cache_path(self.items[index], "text")
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        return None

    def save_text_cache(self, index: int, features: List[torch.Tensor], mask: torch.Tensor) -> None:
        if self.cache_dir is None:
            return
        path = self._cache_path(self.items[index], "text")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": [f.detach().cpu() for f in features],
                "mask": mask.detach().cpu(),
            },
            path,
        )

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        return {
            "index": index,
            "image_path": str(item.image_path),
            "caption": item.caption,
            "height": item.train_height,
            "width": item.train_width,
        }
