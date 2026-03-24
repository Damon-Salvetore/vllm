# SPDX-License-Identifier: Apache-2.0
"""BitBLAS kernel backend for BitNet b1.58 on SM 80-89 (A100/3090/4090).

Provides the same interface as bitnet_kernels.py (TileLang backend):
- get_prefill_kernel(M, N, K) -> (kernel_callable, padded_m)
- precompile_all_kernels()
- transform_weight_bitblas(weight_int8, N, K)

BitBLAS handles dynamic M at runtime (no M-padding needed), so kernels
are cached per (N, K) pair only — 5 instances vs TileLang's 80.

NOTE: BitBLAS was archived by Microsoft on 2026-02-24. It supports
SM 80-89 but NOT SM 120+. This backend is experimental and may not
receive bug fixes upstream.

Requires: pip install bitblas
"""

import functools
import logging
import time

import torch

logger = logging.getLogger(__name__)

# NK pairs for Qwen3.5-27B (same as bitnet_kernels.py)
_NK_PAIRS = [
    (5120, 6144),
    (5120, 17408),
    (14336, 5120),
    (16384, 5120),
    (34816, 5120),
]

_kernels_precompiled = False


def _get_bitblas_target() -> str:
    """Detect GPU target string for BitBLAS.

    Returns a target string like 'cuda' that BitBLAS uses to
    select the appropriate code generation backend.
    """
    # BitBLAS auto-detects from the current CUDA device.
    # Returning "cuda" uses the default device.
    return "cuda"


@functools.lru_cache(maxsize=64)
def _get_matmul(N: int, K: int):
    """Get or create a cached BitBLAS Matmul instance for (N, K).

    BitBLAS Matmul handles M dynamically at runtime, so we only
    need one instance per (N, K) pair.

    Args:
        N: Output features (weight rows)
        K: Input features (weight cols, before packing)

    Returns:
        BitBLAS Matmul instance ready for inference.
    """
    from bitblas import Matmul, MatmulConfig

    config = MatmulConfig(
        N=N,
        K=K,
        A_dtype="int8",
        W_dtype="int2",
        accum_dtype="int32",
        out_dtype="int32",
        layout="nt",  # weight is [N, K], transposed access
    )
    target = _get_bitblas_target()
    matmul = Matmul(config, target=target, enable_tuning=False)
    logger.debug("Created BitBLAS Matmul N=%d K=%d", N, K)
    return matmul


def transform_weight_bitblas(
    weight_int8: torch.Tensor, N: int, K: int
) -> torch.Tensor:
    """Transform ternary weight from int8 to BitBLAS packed format.

    BitBLAS uses its own interleaved packing layout that differs from
    the sequential int2 packing used by TileLang. This function takes
    unpacked int8 ternary weights and repacks them.

    Args:
        weight_int8: [N, K] int8 with values in {-1, 0, 1}
        N: Output features
        K: Input features

    Returns:
        BitBLAS-format packed weight tensor (shape depends on BitBLAS internals)
    """
    matmul = _get_matmul(N, K)
    return matmul.transform_weight(weight_int8)


def get_prefill_kernel(M: int, N: int, K: int):
    """Get BitBLAS GEMM kernel for given dimensions.

    Interface matches bitnet_kernels.get_prefill_kernel() so the caller
    in bitnet.py can use either backend transparently.

    Args:
        M: Batch tokens (actual, no padding needed)
        N: Output features
        K: Input features

    Returns:
        (kernel_callable, padded_m) where:
        - kernel_callable: fn(x_int8 [M, K], qweight [bitblas_packed]) -> [M, N] int32
        - padded_m: always == M (BitBLAS handles dynamic M natively)
    """
    matmul = _get_matmul(N, K)

    def kernel(x_int8: torch.Tensor, qweight: torch.Tensor) -> torch.Tensor:
        return matmul(x_int8, qweight)

    return kernel, M


def precompile_all_kernels(max_m: int | None = None) -> None:
    """Pre-create BitBLAS Matmul instances for all (N, K) pairs.

    BitBLAS handles M dynamically, so only (N, K) pairs need pre-creation.
    This creates 5 kernel instances (vs TileLang's 80).

    Args:
        max_m: Ignored (kept for interface compatibility with TileLang).
    """
    total = len(_NK_PAIRS)
    logger.info("Pre-creating %d BitBLAS kernels...", total)
    t0 = time.time()
    compiled = 0
    for N, K in _NK_PAIRS:
        try:
            _get_matmul(N, K)
            compiled += 1
        except Exception as e:
            logger.warning(
                "Failed to create BitBLAS kernel N=%d K=%d: %s", N, K, e
            )
    elapsed = time.time() - t0
    logger.info(
        "Created %d/%d BitBLAS kernels in %.1fs", compiled, total, elapsed
    )
