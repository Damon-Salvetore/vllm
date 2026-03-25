# SPDX-License-Identifier: Apache-2.0
"""BitNet kernel backend auto-selection based on GPU SM version.

Routes between TileLang (SM >= 120) and BitBLAS (SM 80-89) at runtime.
Falls back to pure torch when no optimized kernel is available.

Override via env: BITNET_KERNEL_BACKEND=tilelang|bitblas|torch
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

BACKEND_TILELANG = "tilelang"
BACKEND_BITBLAS = "bitblas"
BACKEND_TORCH = "torch"

_selected_backend: str | None = None


def detect_backend() -> str:
    """Auto-detect the best kernel backend for current GPU.

    Priority:
    1. BITNET_KERNEL_BACKEND env var (explicit override)
    2. SM >= 120 → TileLang
    3. SM 80-89 → BitBLAS (if importable)
    4. torch fallback
    """
    global _selected_backend
    if _selected_backend is not None:
        return _selected_backend

    env_backend = os.environ.get("BITNET_KERNEL_BACKEND")
    if env_backend:
        valid = {BACKEND_TILELANG, BACKEND_BITBLAS, BACKEND_TORCH}
        if env_backend not in valid:
            raise ValueError(
                f"BITNET_KERNEL_BACKEND={env_backend!r} is invalid. "
                f"Choose from: {', '.join(sorted(valid))}"
            )
        # Verify the requested backend is actually importable
        if env_backend == BACKEND_TILELANG:
            try:
                import tilelang  # noqa: F401
            except ImportError:
                raise ImportError(
                    "BITNET_KERNEL_BACKEND=tilelang but tilelang is not "
                    "installed. Install with: pip install tilelang"
                )
        elif env_backend == BACKEND_BITBLAS:
            try:
                import bitblas  # noqa: F401
            except ImportError:
                raise ImportError(
                    "BITNET_KERNEL_BACKEND=bitblas but bitblas is not "
                    "installed. Install with: pip install bitblas"
                )
        _selected_backend = env_backend
        logger.info("BitNet kernel backend (env override): %s", _selected_backend)
        return _selected_backend

    if not torch.cuda.is_available():
        _selected_backend = BACKEND_TORCH
        logger.info("BitNet kernel backend: %s (no CUDA)", _selected_backend)
        return _selected_backend

    capability = torch.cuda.get_device_capability()
    sm = capability[0] * 10 + capability[1]

    if sm >= 120:
        _selected_backend = BACKEND_TILELANG
    elif sm >= 80:
        # SM 80 (A100), 86 (3090), 89 (4090) — try BitBLAS
        try:
            import bitblas  # noqa: F401
            _selected_backend = BACKEND_BITBLAS
        except ImportError:
            logger.warning(
                "SM %d: BitBLAS not installed, using torch fallback. "
                "Install with: pip install bitblas",
                sm,
            )
            _selected_backend = BACKEND_TORCH
    else:
        logger.warning(
            "SM %d: no optimized BitNet kernel available, using torch fallback",
            sm,
        )
        _selected_backend = BACKEND_TORCH

    logger.info("BitNet kernel backend: %s (SM %d)", _selected_backend, sm)
    return _selected_backend


def get_kernel_module():
    """Import and return the appropriate kernel module.

    Returns None if no optimized backend is available (caller should
    use the torch fallback path).
    """
    backend = detect_backend()
    if backend == BACKEND_TILELANG:
        from vllm.model_executor.layers.quantization import bitnet_kernels
        return bitnet_kernels
    elif backend == BACKEND_BITBLAS:
        from vllm.model_executor.layers.quantization import (
            bitnet_kernels_bitblas,
        )
        return bitnet_kernels_bitblas
    return None
