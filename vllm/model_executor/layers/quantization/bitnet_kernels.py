# SPDX-License-Identifier: Apache-2.0
"""TileLang kernel wrappers for BitNet b1.58 INT2 x INT8 computation.

Contains:
- activation_quant_int8: per-token symmetric INT8 quantization
- pack_int2_weights: ternary {-1,0,1} -> int2 packed (4 per byte)
- build_prefill_kernel: TileLang GEMM for M>1 (prefill)
- build_decode_kernel: TileLang GEMV for M=1 (decode)
- get_prefill_kernel / get_decode_kernel: LRU-cached kernel factories
- prepare_decode_weight: convert sequential packing to interleaved format
- precompile_all_kernels: pre-compile all (M, N, K) combos at startup
"""

import functools
import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Autotune best configs from RTX 5080 benchmark
# Key: (M, N, K) -> (block_M, block_N, block_K, num_stages, threads)
#
# Qwen3.5-27B vLLM-merged layer dimensions:
#   qkv_proj:     N=14336, K=5120  (×16 layers)
#   o_proj:        N=5120,  K=6144  (×16 layers)
#   in_proj_qkvz: N=16384, K=5120  (×48 layers)
#   out_proj:      N=5120,  K=6144  (×48 layers)
#   gate_up_proj: N=34816, K=5120  (×64 layers)
#   down_proj:     N=5120,  K=17408 (×64 layers)
# ---------------------------------------------------------------------------
AUTOTUNE_BEST_CONFIGS: dict[tuple[int, int, int], tuple[int, int, int, int, int]] = {
    # --- M=1,2,4,8 (decode, block_M=16: all share same config per NK) ---
    # Tuned on RTX 5080, 2026-03-18
    (1, 5120, 6144):    (16, 64, 256, 2, 128),   # o_proj/out_proj 13.6μs
    (1, 5120, 17408):   (16, 64, 256, 2, 128),   # down_proj 30.1μs
    (1, 14336, 5120):   (16, 128, 128, 2, 128),  # qkv_proj 19.8μs
    (1, 16384, 5120):   (16, 128, 128, 2, 128),  # in_proj_qkvz 20.3μs
    (1, 34816, 5120):   (16, 256, 128, 2, 128),  # gate_up_proj 32.1μs
    # M=2: same as M=1 (block_M=16 covers both)
    (2, 5120, 6144):    (16, 64, 256, 2, 128),
    (2, 5120, 17408):   (16, 64, 256, 2, 128),
    (2, 14336, 5120):   (16, 128, 128, 2, 128),
    (2, 16384, 5120):   (16, 128, 128, 2, 128),
    (2, 34816, 5120):   (16, 256, 128, 2, 128),
    # M=4: same as M=1
    (4, 5120, 6144):    (16, 64, 256, 2, 128),
    (4, 5120, 17408):   (16, 64, 256, 2, 128),
    (4, 14336, 5120):   (16, 128, 128, 2, 128),
    (4, 16384, 5120):   (16, 128, 128, 2, 128),
    (4, 34816, 5120):   (16, 256, 128, 2, 128),
    # M=8: same as M=1
    (8, 5120, 6144):    (16, 64, 256, 2, 128),
    (8, 5120, 17408):   (16, 64, 256, 2, 128),
    (8, 14336, 5120):   (16, 128, 128, 2, 128),
    (8, 16384, 5120):   (16, 128, 128, 2, 128),
    (8, 34816, 5120):   (16, 256, 128, 2, 128),
    # --- M=16 (prefill) ---
    (16, 5120, 6144):   (32, 64, 256, 2, 128),
    (16, 5120, 17408):  (32, 64, 256, 2, 128),
    (16, 14336, 5120):  (32, 64, 128, 2, 128),
    (16, 16384, 5120):  (32, 64, 128, 2, 128),
    (16, 34816, 5120):  (32, 128, 128, 2, 128),
    # --- M=32 (tuned RTX 5080, 2026-03-18) ---
    (32, 5120, 6144):   (16, 64, 256, 2, 128),   # o_proj 15.8μs 127.6 TOPS
    (32, 5120, 17408):  (32, 64, 256, 2, 128),   # down_proj 38.6μs 147.8 TOPS
    (32, 14336, 5120):  (16, 128, 128, 2, 256),  # qkv_proj 30.6μs 153.5 TOPS
    (32, 16384, 5120):  (32, 256, 128, 2, 256),  # in_proj_qkvz 32.5μs 165.3 TOPS
    (32, 34816, 5120):  (32, 64, 128, 2, 128),   # gate_up_proj 56.9μs 200.6 TOPS
    # --- M=64 (tuned RTX 5080, 2026-03-18) ---
    (64, 5120, 6144):   (64, 64, 256, 2, 128),   # o_proj 21.2μs 189.6 TOPS
    (64, 5120, 17408):  (64, 64, 256, 2, 128),   # down_proj 48.5μs 235.1 TOPS
    (64, 14336, 5120):  (64, 64, 128, 2, 128),   # qkv_proj 39.0μs 240.8 TOPS
    (64, 16384, 5120):  (64, 256, 256, 2, 256),  # in_proj_qkvz 46.1μs 232.8 TOPS
    (64, 34816, 5120):  (64, 64, 128, 2, 128),   # gate_up_proj 81.5μs 279.9 TOPS
    # --- M=128 ---
    (128, 5120, 6144):  (128, 64, 128, 2, 256),
    (128, 5120, 17408): (128, 64, 128, 2, 256),
    (128, 14336, 5120): (64, 64, 128, 2, 128),
    (128, 16384, 5120): (64, 64, 128, 2, 128),
    (128, 34816, 5120): (64, 64, 128, 2, 128),
    # --- M=256 (tuned RTX 5080, 2026-03-18) ---
    (256, 5120, 6144):  (128, 128, 256, 2, 256),  # o_proj 50.6μs 318.3 TOPS
    (256, 5120, 17408): (128, 128, 256, 2, 256),  # down_proj 129.0μs 353.7 TOPS
    (256, 14336, 5120): (64, 64, 128, 2, 128),    # qkv_proj 122.5μs 306.7 TOPS
    (256, 16384, 5120): (128, 64, 128, 2, 128),   # in_proj_qkvz 140.7μs 305.3 TOPS
    (256, 34816, 5120): (128, 64, 128, 2, 128),   # gate_up_proj 260.7μs 350.1 TOPS
    # --- M=512 (tuned RTX 5080, 2026-03-18) ---
    (512, 5120, 6144):  (128, 128, 256, 2, 256),  # o_proj 97.7μs 329.6 TOPS
    (512, 5120, 17408): (128, 128, 256, 2, 256),  # down_proj 255.2μs 357.6 TOPS
    (512, 14336, 5120): (128, 64, 128, 2, 128),   # qkv_proj 218.8μs 343.6 TOPS
    (512, 16384, 5120): (128, 64, 128, 2, 128),   # in_proj_qkvz 255.5μs 336.2 TOPS
    (512, 34816, 5120): (128, 64, 128, 2, 128),   # gate_up_proj 510.7μs 357.4 TOPS
    # --- M=1024 ---
    (1024, 5120, 6144):  (128, 64, 128, 2, 128),
    (1024, 5120, 17408): (128, 128, 256, 2, 256),
    (1024, 14336, 5120): (128, 64, 128, 2, 128),
    (1024, 16384, 5120): (128, 64, 128, 2, 128),
    (1024, 34816, 5120): (128, 64, 128, 2, 128),
    # --- M=2048 (tuned RTX 5080, 2026-03-18) ---
    (2048, 5120, 6144):  (128, 64, 128, 2, 128),   # o_proj 372.2μs 346.2 TOPS
    (2048, 5120, 17408): (128, 128, 256, 2, 256),   # down_proj 1857.9μs 196.5 TOPS
    (2048, 14336, 5120): (128, 64, 128, 2, 128),    # qkv_proj 835.0μs 360.1 TOPS
    (2048, 16384, 5120): (128, 64, 128, 2, 128),    # in_proj_qkvz 954.1μs 360.1 TOPS
    (2048, 34816, 5120): (64, 128, 256, 2, 128),    # gate_up_proj 2115.5μs 345.1 TOPS
    # --- M=4096 ---
    (4096, 5120, 6144):  (128, 64, 128, 2, 128),
    (4096, 5120, 17408): (128, 128, 256, 2, 256),
    (4096, 14336, 5120): (128, 64, 128, 2, 128),
    (4096, 16384, 5120): (128, 64, 128, 2, 128),
    (4096, 34816, 5120): (128, 128, 256, 2, 256),
    # --- M=8192 (tuned RTX 5080, 2026-03-18) ---
    (8192, 5120, 6144):  (128, 64, 128, 2, 128),    # o_proj 1412.5μs 364.9 TOPS
    (8192, 5120, 17408): (128, 128, 256, 2, 256),   # down_proj 3776.0μs 386.7 TOPS
    (8192, 14336, 5120): (128, 128, 256, 2, 256),   # qkv_proj 3291.0μs 365.4 TOPS
    (8192, 16384, 5120): (128, 128, 256, 2, 256),   # in_proj_qkvz 3754.5μs 366.1 TOPS
    (8192, 34816, 5120): (128, 128, 256, 2, 256),   # gate_up_proj 7582.6μs 385.2 TOPS
    # --- M=16384 ---
    (16384, 5120, 6144):  (128, 64, 128, 2, 128),
    (16384, 5120, 17408): (128, 128, 256, 2, 256),
    (16384, 14336, 5120): (128, 64, 128, 2, 128),
    (16384, 16384, 5120): (128, 64, 128, 2, 128),
    (16384, 34816, 5120): (128, 128, 256, 2, 256),
    # --- M=32768 (tuned RTX 5080, 2026-03-18; gate_up_proj OOM, use default) ---
    (32768, 5120, 6144):  (128, 128, 256, 2, 256),  # o_proj 5457.9μs 377.7 TOPS
    (32768, 5120, 17408): (128, 128, 256, 2, 256),  # down_proj 14543.9μs 401.6 TOPS
    (32768, 14336, 5120): (128, 128, 256, 2, 256),  # qkv_proj 12424.5μs 387.2 TOPS
    (32768, 16384, 5120): (128, 128, 256, 2, 256),  # in_proj_qkvz 14240.7μs 386.0 TOPS
    (32768, 34816, 5120): (128, 128, 256, 2, 256),  # gate_up_proj (default, autotune OOM)
}

# M-padding: power-of-2 sequence covering all config table M values.
# _round_up_m() maps any actual M to the nearest value in this list.
_PADDED_M_VALUES = sorted(set(m for m, _, _ in AUTOTUNE_BEST_CONFIGS))

# Sorted M values in config table for round-up matching
_CONFIG_M_VALUES = _PADDED_M_VALUES

# NK pairs for Qwen3.5-27B (used by precompile_all_kernels)
_NK_PAIRS = [
    (5120, 6144), (5120, 17408), (14336, 5120),
    (16384, 5120), (34816, 5120),
]

# Default config when no autotune entry exists
DEFAULT_PREFILL_CONFIG = (128, 64, 128, 2, 128)


# ---------------------------------------------------------------------------
# Activation quantization
# ---------------------------------------------------------------------------

def activation_quant_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric INT8 quantization.

    Args:
        x: [M, K] activation tensor (bf16/fp16).

    Returns:
        x_int8: [M, K] int8 quantized.
        scale: [M, 1] fp32, where scale = 127 / max(|x|) per row.
    """
    max_vals = x.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-5)
    scale = 127.0 / max_vals  # [M, 1]
    x_int8 = (x * scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale.to(torch.float32)


# ---------------------------------------------------------------------------
# Weight packing
# ---------------------------------------------------------------------------

def pack_int2_weights(weight: torch.Tensor) -> torch.Tensor:
    """Pack int8 ternary {-1,0,1} into int2 (4 values per byte).

    Packing: values offset by +2, so {-1,0,1} -> {1,2,3}.
    4 values packed: packed = v0 | (v1<<2) | (v2<<4) | (v3<<6).

    Args:
        weight: [N, K] int8 with values in {-1, 0, 1}.

    Returns:
        packed: [N, K//4] int8.
    """
    assert weight.dim() == 2
    N, K = weight.shape
    assert K % 4 == 0

    offset = (weight.to(torch.int32) + 2).to(torch.uint8)
    offset = offset.reshape(N, K // 4, 4)
    packed = (
        offset[:, :, 0]
        | (offset[:, :, 1] << 2)
        | (offset[:, :, 2] << 4)
        | (offset[:, :, 3] << 6)
    )
    return packed.to(torch.int8)


# ---------------------------------------------------------------------------
# TileLang prefill kernel (GEMM, M > 1)
# ---------------------------------------------------------------------------

def _tir_u8_to_i2_to_i8(val, pos):
    """TIR function: extract 2-bit value from packed byte and cast to int8."""
    from tvm import tir
    nbit = 2
    mask = tir.const((1 << nbit) - 1, "uint8")
    val_u8 = tir.reinterpret("uint8", val)
    shift = pos.astype("uint8") * tir.const(nbit, "uint8")
    extracted = (val_u8 >> shift) & mask
    return extracted.astype("int8")


def build_prefill_kernel(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    """Build TileLang prefill GEMM kernel (INT8 x INT2 -> INT32).

    Uses simple sequential bit packing (no interleave_weight).
    Returns a compiled kernel callable: kernel(A_int8, qw) -> C_int32.
    """
    import tilelang
    import tilelang.language as T

    num_elems_per_byte = 4
    in_dtype = T.int8
    out_dtype = T.int32
    accum_dtype = T.int32
    storage_dtype = T.int8

    @tilelang.jit(out_idx=[2])
    def kernel_func():

        @T.prim_func
        def main(
            A: T.Tensor((M, K), in_dtype),
            B: T.Tensor((N, K // num_elems_per_byte), storage_dtype),
            C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(
                T.ceildiv(N, block_N),
                T.ceildiv(M, block_M),
                threads=threads,
            ) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), in_dtype)
                B_shared = T.alloc_shared(
                    (block_N, block_K // num_elems_per_byte), storage_dtype
                )
                B_local = T.alloc_fragment(
                    (block_N, block_K // num_elems_per_byte), storage_dtype
                )
                B_dequant_local = T.alloc_fragment(
                    (block_N, block_K), in_dtype
                )
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), out_dtype)

                T.use_swizzle(panel_size=10)

                T.clear(C_local)
                for k in T.Pipelined(K // block_K, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(
                        B[bx * block_N, k * block_K // num_elems_per_byte],
                        B_shared,
                    )
                    T.copy(B_shared, B_local)
                    for i, j in T.Parallel(block_N, block_K):
                        B_dequant_local[i, j] = _tir_u8_to_i2_to_i8(
                            B_local[i, j // num_elems_per_byte],
                            j % num_elems_per_byte,
                        )
                    T.gemm(
                        A_shared, B_dequant_local, C_local, transpose_B=True
                    )
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return main

    return kernel_func()


# ---------------------------------------------------------------------------
# TileLang decode kernel (GEMV, M = 1)
# ---------------------------------------------------------------------------

# CUDA source for fast int2 -> int8 dequantization
_DECODE_I2_TO_I8_SOURCE = """template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2b, T2 *_i8s, const int N = 16)
{
    uint *i8s = reinterpret_cast<uint *>(_i8s);
    uint const i2b = *reinterpret_cast<uint *>(_i2b);
    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x03030303;
    static constexpr uint I8s_MAGIC_NUM = 0x00000000;
    static constexpr uint MEDIAN_NUM = 0x02020202;
#pragma unroll
    for (int i = 0; i < (N / 4); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(i8s[i])
                     : "r"(i2b >> (2 * i)), "n"(BOTTOM_MASK), "n"(I8s_MAGIC_NUM), "n"(immLut));
        i8s[i] = __vsub4(i8s[i], MEDIAN_NUM);
    }
}
template <typename T1, typename T2>
__device__ void decode_i2u_to_i8s(T1 *_i2b, T2 *_i8s, const int N = 16)
{
    uint *i8s = reinterpret_cast<uint *>(_i8s);
    uint const i2b = *reinterpret_cast<uint *>(_i2b);
    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x03030303;
    static constexpr uint I8s_MAGIC_NUM = 0x00000000;
#pragma unroll
    for (int i = 0; i < (N / 4); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(i8s[i])
                     : "r"(i2b >> (2 * i)), "n"(BOTTOM_MASK), "n"(I8s_MAGIC_NUM), "n"(immLut));
    }
}
"""


def build_decode_kernel(
    M: int,
    N: int,
    K: int,
    n_partition: int = 4,
    reduce_thread: int = 32,
):
    """Build TileLang decode GEMV kernel (INT8 x INT2 -> INT32).

    Uses interleaved weight packing (general_compress + interleave_weight).
    Returns: compiled kernel callable: kernel(A_int8, qw_interleaved, C_int32).
    """
    import tilelang
    import tilelang.language as T
    from tilelang import tvm as tvm
    from tvm import DataType

    in_dtype = T.int8
    out_dtype = T.int32
    accum_dtype = T.int32
    storage_nbit = 8
    num_bits = 2
    num_elems_per_byte = 4
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    micro_size_k = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
    micro_size_k_compressed = micro_size_k // num_elems_per_byte
    storage_dtype = T.int8
    block_K = reduce_thread * micro_size_k

    dp4a_size = 4

    @T.prim_func
    def program(
        A: T.Buffer((M, K), in_dtype),
        B: T.Buffer((N, K // storage_nbit * num_bits), storage_dtype),
        C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, n_partition),
            M,
            threads=(reduce_thread, n_partition),
        ) as (bx, by):
            A_local = T.alloc_local((micro_size_k,), in_dtype)
            B_quant_local = T.alloc_local(
                [micro_size_k_compressed], storage_dtype
            )
            B_dequantize_local = T.alloc_local([micro_size_k], in_dtype)
            accum_res = T.alloc_local((1,), accum_dtype)
            reduced_accum_res = T.alloc_local((1,), accum_dtype)

            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")

            T.import_source(_DECODE_I2_TO_I8_SOURCE)

            T.clear(accum_res)
            for ko in T.serial(T.ceildiv(K, block_K)):
                for v in T.vectorized(micro_size_k):
                    A_local[v] = A[
                        by, ko * block_K + kr * micro_size_k + v
                    ]

                for v in T.vectorized(micro_size_k_compressed):
                    B_quant_local[v] = B[
                        bx * n_partition + ni,
                        ko * (reduce_thread * micro_size_k_compressed)
                        + kr * micro_size_k_compressed
                        + v,
                    ]

                T.call_extern(
                    "handle",
                    "decode_i2u_to_i8s",
                    T.access_ptr(B_quant_local, "r"),
                    T.access_ptr(B_dequantize_local, "w"),
                )

                for ki in T.serial(micro_size_k // dp4a_size):
                    T.dp4a(
                        A_local[ki * dp4a_size],
                        B_dequantize_local[ki * dp4a_size],
                        accum_res[0],
                    )

            with T.attr(
                T.comm_reducer(
                    lambda x, y: x + y, [T.cast(0, accum_dtype)]
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accum_res[0],
                        True,
                        reduced_accum_res[0],
                        kr,
                        dtype="handle",
                    )
                )
            if kr == 0:
                C[by, bx * n_partition + ni] = reduced_accum_res[0]

    kernel = tilelang.compile(program)
    return kernel


# ---------------------------------------------------------------------------
# Weight format conversion for decode kernel
# ---------------------------------------------------------------------------

def general_compress(
    lowprecision_weight: np.ndarray,
    source_bits: int = 2,
    storage_dtype: type = np.int8,
) -> np.ndarray:
    """Compress low-bit weight array into packed bytes.

    Args:
        lowprecision_weight: [N, K] with values in [0, 2^source_bits).
        source_bits: bits per element (2 for ternary).

    Returns:
        Packed array [N, K // elems_per_byte].
    """
    elems_per_byte = 8 // source_bits
    if lowprecision_weight.dtype == np.float16:
        lowprecision_weight = lowprecision_weight.astype(np.int8)
    int8_weight = np.zeros(
        (
            *lowprecision_weight.shape[:-1],
            lowprecision_weight.shape[-1] // elems_per_byte,
        ),
        dtype=np.int8,
    )
    for j in range(lowprecision_weight.shape[-1] // elems_per_byte):
        for k in range(elems_per_byte):
            int8_weight[:, j] |= (
                lowprecision_weight[:, j * elems_per_byte + k]
                << (source_bits * k)
            )
    return int8_weight.view(storage_dtype)


def interleave_weight(
    qweight: np.ndarray,
    nbits: int = 2,
    target_dtype: Any = None,
) -> np.ndarray:
    """Interleave packed weight for TileLang decode kernel memory access.

    Args:
        qweight: compressed weight from general_compress().
        nbits: bits per element.
        target_dtype: tilelang dtype (T.int8 or T.float16).

    Returns:
        Interleaved weight array.
    """
    import tilelang.language as T

    if target_dtype is None:
        target_dtype = T.int8

    assert target_dtype in [T.float16, T.int8]
    qweight = qweight.view(np.int32)
    new_qweight = np.zeros_like(qweight)
    bits_stride = 8 if target_dtype == T.int8 else 16
    mask = (1 << nbits) - 1
    num_groups = 32 // bits_stride
    elems_per_group = bits_stride // nbits
    for i in range(num_groups):
        for j in range(elems_per_group):
            offset = i * elems_per_group + j
            shift = (offset % num_groups) * bits_stride + (
                offset // num_groups
            ) * nbits
            new_qweight |= ((qweight >> (nbits * offset)) & mask) << shift

    if nbits == 2 and target_dtype == T.float16:
        n8_weight = new_qweight & np.int32(0xFF0000FF)
        n8_weight |= ((new_qweight & np.int32(0x0000FF00)) >> 8) << 16
        n8_weight |= ((new_qweight & np.int32(0x00FF0000)) >> 16) << 8
        return n8_weight.view(np.int8)

    return new_qweight.view(np.int8)


def prepare_decode_weight(
    qweight: torch.Tensor, N: int, K: int
) -> torch.Tensor:
    """Convert sequentially-packed int2 weights to interleaved decode format.

    Args:
        qweight: [N, K//4] int8 with simple sequential packing.
        N: output dimension.
        K: input dimension (unpacked).

    Returns:
        qweight_decode: [N, K//4] int8 in interleaved format for decode kernel.
    """
    import tilelang.language as T

    # Unpack to raw unsigned 2-bit values
    p = qweight.cpu().numpy().view(np.uint8)
    v0 = p & 0x03
    v1 = (p >> 2) & 0x03
    v2 = (p >> 4) & 0x03
    v3 = (p >> 6) & 0x03
    raw = np.stack([v0, v1, v2, v3], axis=-1).reshape(N, K).astype(np.int8)

    # Re-compress with general_compress + interleave
    qw = general_compress(raw, source_bits=2, storage_dtype=np.int8)
    qw = interleave_weight(qw, nbits=2, target_dtype=T.int8)
    return torch.from_numpy(qw.copy()).to(device=qweight.device)


# ---------------------------------------------------------------------------
# Cached kernel factories
# ---------------------------------------------------------------------------

def _round_up_m(M: int) -> int:
    """Round M up to the next value in the config table.

    E.g. M=3 -> 4, M=5 -> 8, M=9 -> 16.
    If M exceeds all entries, return M unchanged (will use default config).
    """
    for m_val in _CONFIG_M_VALUES:
        if m_val >= M:
            return m_val
    return M


@functools.lru_cache(maxsize=128)
def get_prefill_kernel(M: int, N: int, K: int):
    """Get or build a cached prefill kernel for given dimensions.

    Kernel is compiled with **padded M** (rounded up to next config value).
    Caller must zero-pad input to padded_M rows and slice output.
    Returns: (kernel, padded_m)
    """
    padded_m = _round_up_m(M)
    config_key = (padded_m, N, K)
    if config_key in AUTOTUNE_BEST_CONFIGS:
        bM, bN, bK, ns, thr = AUTOTUNE_BEST_CONFIGS[config_key]
    else:
        bM, bN, bK, ns, thr = DEFAULT_PREFILL_CONFIG
    kernel = build_prefill_kernel(
        padded_m, N, K,
        block_M=bM, block_N=bN, block_K=bK,
        num_stages=ns, threads=thr,
    )
    return kernel, padded_m


@functools.lru_cache(maxsize=64)
def get_decode_kernel(M: int, N: int, K: int):
    """Get or build a cached decode kernel for given dimensions."""
    return build_decode_kernel(M, N, K)


# ---------------------------------------------------------------------------
# Kernel precompilation
# ---------------------------------------------------------------------------

_kernels_precompiled = False


def precompile_all_kernels(
    max_m: int | None = None,
) -> None:
    """Pre-compile TileLang kernels for all (M, N, K) combinations.

    Called once at vLLM startup to avoid runtime compilation latency.
    Only compiles for M values up to max_m (default: all padded values).
    Set BITNET_PRECOMPILE_MAX_M env var to limit (e.g. "1024").
    """
    import os
    import time

    if max_m is None:
        env_max = os.environ.get("BITNET_PRECOMPILE_MAX_M")
        max_m = int(env_max) if env_max else max(_PADDED_M_VALUES)

    m_values = [m for m in _PADDED_M_VALUES if m <= max_m]
    total = len(m_values) * len(_NK_PAIRS)
    logger.info(
        "Pre-compiling %d TileLang kernels (M up to %d)...", total, max_m
    )
    t0 = time.time()
    compiled = 0
    for m_val in m_values:
        for N, K in _NK_PAIRS:
            try:
                get_prefill_kernel(m_val, N, K)
                compiled += 1
            except Exception as e:
                logger.warning(
                    "Failed to precompile kernel M=%d N=%d K=%d: %s",
                    m_val, N, K, e,
                )
    elapsed = time.time() - t0
    logger.info(
        "Pre-compiled %d/%d kernels in %.1fs", compiled, total, elapsed
    )
