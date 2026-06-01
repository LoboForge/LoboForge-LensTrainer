"""Sanitize CUDA library paths before importing torch.

Cursor and some AppImage shells prepend ``LD_LIBRARY_PATH`` entries that can
shadow PyTorch's bundled ``libcublasLt.so.12``. The first loaded symbol wins,
which often surfaces as::

    Invalid handle. Cannot load symbol cublasLtGetVersion
    Aborted (core dumped)

Call :func:`configure_cuda_libraries` at process startup (before ``import torch``).
"""

from __future__ import annotations

import ctypes
import glob
import os
import site
import sys
import warnings
from pathlib import Path

_CONFIGURED = False


def _suppress_known_torch_warnings() -> None:
    """PyTorch 2.x warns when the legacy ``pynvml`` package is installed."""
    warnings.filterwarnings(
        "ignore",
        message="The pynvml package is deprecated.*",
        category=FutureWarning,
    )


def _nvidia_lib_dirs() -> list[Path]:
    dirs: list[Path] = []
    candidates = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(user_site)

    for sp in candidates:
        nvidia_root = Path(sp) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir():
                dirs.append(lib_dir)
    return dirs


def _sanitize_ld_library_path(nvidia_dirs: list[Path]) -> None:
    prepend = ":".join(str(path) for path in nvidia_dirs)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in existing.split(":") if part]

    blocked = (
        ".mount_Cursor",
        "/tmp/.mount_",
        "appimage",
    )
    cleaned = [part for part in parts if not any(token in part for token in blocked)]

    merged: list[str] = []
    for part in [prepend, *cleaned]:
        if part and part not in merged:
            merged.append(part)
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


def _preload_shared_objects(nvidia_dirs: list[Path]) -> None:
    patterns = (
        "libcublasLt.so*",
        "libcublas.so*",
        "libcudnn.so*",
    )
    for lib_dir in nvidia_dirs:
        for pattern in patterns:
            for lib_path in sorted(glob.glob(str(lib_dir / pattern))):
                if not os.path.isfile(lib_path):
                    continue
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue


def configure_cuda_libraries() -> None:
    global _CONFIGURED
    _suppress_known_torch_warnings()
    if _CONFIGURED or sys.platform != "linux":
        return
    nvidia_dirs = _nvidia_lib_dirs()
    if nvidia_dirs:
        _sanitize_ld_library_path(nvidia_dirs)
        _preload_shared_objects(nvidia_dirs)
    _CONFIGURED = True
