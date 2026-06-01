"""Dataset with optional latent and text-embedding caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from lens_trainer.config import DatasetConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class DatasetItem:
    image_path: Path
    caption: str


def discover_items(folder: Path, caption_ext: str) -> List[DatasetItem]:
    items: List[DatasetItem] = []
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
            continue
        items.append(DatasetItem(image_path=image_path, caption=caption))
    if not items:
        raise ValueError(
            f"No image/caption pairs found under {folder}. "
            f"Expected image.jpg + image.{caption_ext} side by side."
        )
    return items


class LensDataset(Dataset):
    def __init__(
        self,
        cfg: DatasetConfig,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.folder = Path(cfg.folder_path)
        self.items = discover_items(self.folder, cfg.caption_ext)
        self.resolution = int(cfg.resolution)
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "folder": str(self.folder),
            "resolution": self.resolution,
            "count": len(self.items),
            "items": [
                {"image": str(it.image_path), "caption": it.caption}
                for it in self.items
            ],
        }
        (self.cache_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def __len__(self) -> int:
        return len(self.items)

    def _cache_key(self, item: DatasetItem, kind: str) -> str:
        payload = f"{kind}|{item.image_path}|{item.caption}|{self.resolution}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, item: DatasetItem, kind: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / kind / f"{self._cache_key(item, kind)}.pt"

    def get_image(self, index: int) -> Image.Image:
        item = self.items[index]
        return Image.open(item.image_path).convert("RGB")

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
            "resolution": self.resolution,
        }
