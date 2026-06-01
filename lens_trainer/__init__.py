"""Lens LoRA trainer package."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["TrainerConfig", "load_config", "LensTrainer"]

if TYPE_CHECKING:
    from lens_trainer.config import TrainerConfig, load_config
    from lens_trainer.trainer import LensTrainer


def __getattr__(name: str):
    if name in {"TrainerConfig", "load_config"}:
        from lens_trainer.config import TrainerConfig, load_config

        return TrainerConfig if name == "TrainerConfig" else load_config
    if name == "LensTrainer":
        from lens_trainer.trainer import LensTrainer

        return LensTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
