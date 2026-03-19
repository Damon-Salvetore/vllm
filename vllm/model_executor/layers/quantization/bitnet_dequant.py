# SPDX-License-Identifier: Apache-2.0
"""Triton fused dequantization + offset correction + bias kernel for BitNet.

Computes:
    output = (gemm_output - row_sum * offset) * scale_a[M,1] * scale_b[1,N] + bias[1,N]

where row_sum[M] = sum_k(x_int8) and offset=2 corrects the {-1,0,1}+2 packing.

Supports INT32 input (from TileLang int2 x int8 kernel) and optional bias.
Adapted from slidesparse basic_dequant_bias_triton.py.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_bias_kernel(
    gemm_output_ptr,
    scale_a_ptr,
    scale_b_ptr,
    bias_ptr,
    row_sum_ptr,
    output_ptr,
    M,
    N,
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INPUT_FP32: tl.constexpr,
    INPUT_INT32: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_OFFSET: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    row_offs = row_start + tl.arange(0, BLOCK_M)
    col_offs = col_start + tl.arange(0, BLOCK_N)

    row_mask = row_offs < M
    col_mask = col_offs < N
    mask_2d = row_mask[:, None] & col_mask[None, :]

    scale_a = tl.load(scale_a_ptr + row_offs, mask=row_mask, other=1.0)
    scale_b = tl.load(scale_b_ptr + col_offs, mask=col_mask, other=1.0)

    gemm_offs = row_offs[:, None] * stride_gm + col_offs[None, :] * stride_gn
    gemm_val = tl.load(gemm_output_ptr + gemm_offs, mask=mask_2d, other=0.0)

    if INPUT_INT32:
        # Offset correction in INT32 domain (before float conversion)
        # to preserve full precision for large K accumulations.
        if HAS_OFFSET:
            rsum = tl.load(row_sum_ptr + row_offs, mask=row_mask, other=0)
            gemm_val = gemm_val - rsum[:, None] * OFFSET
        gemm_val = gemm_val.to(tl.float32)
    elif not INPUT_FP32:
        gemm_val = gemm_val.to(tl.float32)
        if HAS_OFFSET:
            rsum = tl.load(row_sum_ptr + row_offs, mask=row_mask, other=0.0)
            gemm_val = gemm_val - rsum[:, None] * OFFSET
    else:
        if HAS_OFFSET:
            rsum = tl.load(row_sum_ptr + row_offs, mask=row_mask, other=0.0)
            gemm_val = gemm_val - rsum[:, None] * OFFSET

    output_val = gemm_val * scale_a[:, None] * scale_b[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + col_offs, mask=col_mask, other=0.0)
        bias = bias.to(tl.float32)
        output_val = output_val + bias[None, :]

    output_val = output_val.to(tl.bfloat16)

    output_offs = row_offs[:, None] * stride_om + col_offs[None, :] * stride_on
    tl.store(output_ptr + output_offs, output_val, mask=mask_2d)


def _get_best_config(M: int, N: int) -> tuple[int, int, int]:
    """Select (BLOCK_M, BLOCK_N, num_warps) based on matrix size.

    Auto-tuned on RTX 5080 for Qwen3.5-27B BitNet.
    M=1..8: autotune_dequant_m1248.py
    M=16..256: autotune_dequant_all_m.py (run 1)
    M=512..32768: autotune_dequant_all_m.py (run 2)
    """
    if N == 5120:
        if M <= 4:
            return 16, 64, 2
        elif M <= 8:
            return 16, 128, 2
        elif M <= 16:
            return 16, 64, 2
        elif M <= 32:
            return 16, 128, 4
        elif M <= 64:
            return 32, 64, 2
        elif M <= 128:
            return 16, 64, 2
        elif M <= 512:
            return 16, 512, 4
        elif M <= 1024:
            return 128, 256, 8
        elif M <= 16384:
            return 16, 1024, 8
        return 16, 2048, 8
    elif N == 14336:
        if M <= 4:
            return 16, 64, 2
        elif M <= 8:
            return 32, 64, 2
        elif M <= 32:
            return 16, 64, 2
        elif M <= 64:
            return 16, 512, 4
        elif M <= 128:
            return 64, 512, 8
        elif M <= 512:
            return 64, 256, 8
        elif M <= 1024:
            return 16, 1024, 8
        elif M <= 2048:
            return 16, 256, 4
        elif M <= 4096:
            return 16, 1024, 8
        elif M <= 8192:
            return 16, 1024, 8
        elif M <= 16384:
            return 16, 512, 4
        return 16, 2048, 8
    elif N == 16384:
        if M <= 1:
            return 16, 64, 2
        elif M <= 2:
            return 16, 128, 4
        elif M <= 4:
            return 16, 512, 4
        elif M <= 8:
            return 32, 64, 2
        elif M <= 16:
            return 16, 64, 2
        elif M <= 32:
            return 128, 64, 4
        elif M <= 64:
            return 16, 512, 4
        elif M <= 128:
            return 16, 2048, 8
        elif M <= 1024:
            return 16, 1024, 8
        elif M <= 2048:
            return 16, 512, 4
        elif M <= 4096:
            return 32, 1024, 8
        elif M <= 8192:
            return 128, 256, 8
        elif M <= 16384:
            return 16, 1024, 8
        return 32, 1024, 8
    elif N == 34816:
        if M <= 8:
            return 16, 64, 2
        elif M <= 32:
            return 64, 512, 8
        elif M <= 64:
            return 16, 512, 4
        elif M <= 128:
            return 64, 256, 8
        elif M <= 1024:
            return 16, 1024, 8
        elif M <= 2048:
            return 16, 2048, 8
        elif M <= 4096:
            return 16, 2048, 8
        elif M <= 8192:
            return 16, 2048, 8
        elif M <= 16384:
            return 16, 512, 4
        return 16, 2048, 8
    elif N == 248320:
        if M <= 16:
            return 16, 1024, 8
        elif M <= 32:
            return 16, 1024, 8
        elif M <= 64:
            return 64, 256, 8
        elif M <= 128:
            return 64, 256, 8
        return 64, 256, 8
    # Default fallback for unknown N
    if M <= 8:
        return 16, 128, 4
    elif M <= 64:
        return 32, 64, 4
    elif M <= 512:
        return 64, 64, 4
    elif M <= 4096:
        return 64, 128, 8
    return 128, 64, 8


def _prepare_scale(scale: torch.Tensor, size: int) -> torch.Tensor:
    """Ensure scale is 1D float32 contiguous."""
    if scale.numel() == 1:
        scale = scale.view(1).expand(size)
    else:
        scale = scale.view(-1)
    if scale.dtype != torch.float32:
        return scale.contiguous().float()
    return scale.contiguous() if not scale.is_contiguous() else scale


def dequant_bias_triton(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    row_sum: torch.Tensor | None = None,
    packing_offset: int = 2,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Triton fused dequant + offset correction + bias.

    Computes:
        corrected = gemm_output - row_sum * packing_offset   (if row_sum given)
        output = corrected * scale_a[M,1] * scale_b[1,N] + bias[1,N]

    Args:
        gemm_output: [M, N] INT32 / FP32 / BF16.
        scale_a: [M, 1] or [M] FP32.
        scale_b: [1, N] or [N] FP32.
        bias: [N] BF16 or None.
        row_sum: [M] FP32 — sum of x_int8 per row, for offset correction.
        packing_offset: int, the offset used in packing (default 2).
        out_dtype: output dtype (default bf16).

    Returns:
        [M, N] tensor in out_dtype.
    """
    assert gemm_output.is_cuda
    assert gemm_output.is_contiguous()
    assert gemm_output.dtype in (torch.bfloat16, torch.float32, torch.int32)

    M, N = gemm_output.shape
    input_fp32 = gemm_output.dtype == torch.float32
    input_int32 = gemm_output.dtype == torch.int32

    scale_a = _prepare_scale(scale_a, M)
    assert scale_a.shape[0] == M

    scale_b = _prepare_scale(scale_b, N)
    assert scale_b.shape[0] == N

    has_bias = bias is not None
    if has_bias:
        bias = bias.view(-1)
        if bias.dtype != torch.bfloat16:
            bias = bias.to(torch.bfloat16)
        bias = bias.contiguous() if not bias.is_contiguous() else bias
        assert bias.shape[0] == N
    else:
        bias = scale_b  # dummy ptr, won't be loaded

    has_offset = row_sum is not None
    if has_offset:
        row_sum = row_sum.view(-1)
        # When gemm_output is INT32, keep row_sum as INT32 for
        # offset correction in integer domain (better precision).
        # Otherwise convert to float32.
        if input_int32:
            if row_sum.dtype != torch.int32:
                row_sum = row_sum.to(torch.int32)
        else:
            if row_sum.dtype != torch.float32:
                row_sum = row_sum.contiguous().float()
        row_sum = (
            row_sum.contiguous() if not row_sum.is_contiguous() else row_sum
        )
        assert row_sum.shape[0] == M
    else:
        row_sum = scale_a  # dummy ptr, won't be loaded

    output = torch.empty(
        (M, N), dtype=torch.bfloat16, device=gemm_output.device
    )

    block_m, block_n, num_warps = _get_best_config(M, N)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    stride_gm, stride_gn = gemm_output.stride()
    stride_om, stride_on = output.stride()

    _dequant_bias_kernel[grid](
        gemm_output,
        scale_a,
        scale_b,
        bias,
        row_sum,
        output,
        M,
        N,
        stride_gm,
        stride_gn,
        stride_om,
        stride_on,
        OFFSET=packing_offset,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        INPUT_FP32=input_fp32,
        INPUT_INT32=input_int32,
        HAS_BIAS=has_bias,
        HAS_OFFSET=has_offset,
        num_warps=num_warps,
    )

    if out_dtype != torch.bfloat16:
        output = output.to(out_dtype)

    return output
