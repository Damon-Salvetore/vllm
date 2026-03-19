# SPDX-License-Identifier: Apache-2.0
"""BitNet b1.58 ternary quantization for vLLM.

Weights: ternary {-1, 0, 1} stored as int2 packed (4 values per byte).
Activations: per-token symmetric int8 at runtime.
Kernel backend: TileLang int2 x int8 (prefill: GEMM, decode: GEMV).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    PackedvLLMParameter,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods
else:
    QuantizationMethods = str

logger = init_logger(__name__)


class BitNetConfig(QuantizationConfig):
    """Config for BitNet b1.58 ternary quantization.

    Expected quantize_config.json:
    {
        "quant_method": "bitnet",
        "weight_bits": 2,
        "group_size": -1
    }
    """

    def __init__(
        self,
        weight_bits: int = 2,
        group_size: int = -1,
    ) -> None:
        super().__init__()
        if weight_bits != 2:
            raise ValueError(
                f"BitNet only supports 2-bit weights, got {weight_bits}"
            )
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.pack_factor = 8 // self.weight_bits  # = 4

    def __repr__(self) -> str:
        return (
            f"BitNetConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "bitnet"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80  # Ampere+

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BitNetConfig":
        weight_bits = cls.get_from_keys_or(config, ["weight_bits", "bits"], 2)
        group_size = cls.get_from_keys_or(config, ["group_size"], -1)
        return cls(weight_bits=weight_bits, group_size=group_size)

    # Suffixes of module paths that should be ternarized.
    # These correspond to the merged vLLM layer names.
    TERNARIZED_MODULE_SUFFIXES = (
        # self_attn
        "qkv_proj",
        "o_proj",
        # linear_attn (GDN)
        "in_proj_qkvz",
        "out_proj",
        # MLP
        "gate_up_proj",
        "down_proj",
    )

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            # Only quantize layers that were ternarized in the checkpoint.
            # Small linear_attn params (in_proj_a/b) and visual layers are BF16.
            if any(prefix.endswith(s) for s in self.TERNARIZED_MODULE_SUFFIXES):
                return BitNetLinearMethod(self)
            logger.info("BitNet: skipping non-ternarized layer %s", prefix)
            return UnquantizedLinearMethod()
        if isinstance(layer, VocabParallelEmbedding):
            # INT8 per-channel quantized embed_tokens / lm_head
            return BitNetInt8EmbeddingMethod()
        return None


class BitNetLinearMethod(LinearMethodBase):
    """INT2 weight x INT8 activation linear method for BitNet b1.58.

    Weight layout:
        - weight: int2 packed as int8, shape [N, K // 4]
        - weight_scale: per-tensor scale (beta = mean(|W|)), shape [1] bf16
    """

    def __init__(self, quant_config: BitNetConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del output_size
        output_size_per_partition = sum(output_partition_sizes)
        pack_factor = self.quant_config.pack_factor  # 4

        if input_size_per_partition % pack_factor != 0:
            raise ValueError(
                f"input_size_per_partition ({input_size_per_partition}) "
                f"must be divisible by pack_factor ({pack_factor})"
            )

        weight_loader = extra_weight_attrs.get("weight_loader")

        # Packed int2 weight: [N, K // 4] stored as int8
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // pack_factor,
                dtype=torch.int8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=pack_factor,
            weight_loader=weight_loader,
        )

        # Per-channel weight scale: [N] bf16, broadcast from per-tensor scalar
        weight_scale = ChannelQuantScaleParameter(
            data=torch.ones(
                output_size_per_partition, 1,
                dtype=torch.bfloat16,
            ),
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", weight_scale)

        # Store dims for apply()
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # Freeze parameters
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = Parameter(
            layer.weight_scale.data, requires_grad=False
        )

        # Skip decode weight copy to save ~3GB VRAM.
        # M=1 decode will fall through to prefill kernel (GEMM instead of GEMV).
        layer.weight_decode = None

        # Pre-compile all TileLang kernels on first layer load
        import vllm.model_executor.layers.quantization.bitnet_kernels as _bk
        if not _bk._kernels_precompiled:
            _bk._kernels_precompiled = True
            _bk.precompile_all_kernels()

    @torch.compiler.disable
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qweight = layer.weight  # [N, K//4] int8 packed
        weight_scale = layer.weight_scale  # [N, 1] bf16 per-channel
        K = layer.input_size_per_partition
        N = layer.output_size_per_partition

        out_shape = x.shape[:-1] + (N,)
        M = x[..., 0].numel()  # total tokens

        try:
            return self._apply_tilelang(
                layer, x, bias, qweight, weight_scale, K, N, M, out_shape
            )
        except Exception as e:
            logger.warning(
                "BitNet TileLang kernel failed (%s), falling back to torch",
                e, exc_info=True,
            )
            return self._apply_fallback(
                x, bias, qweight, weight_scale, K, N, out_shape
            )

    def _apply_tilelang(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        qweight: torch.Tensor,
        weight_scale: torch.Tensor,
        K: int,
        N: int,
        M: int,
        out_shape: tuple,
    ) -> torch.Tensor:
        from vllm._custom_ops import scaled_int8_quant
        from vllm.model_executor.layers.quantization.bitnet_kernels import (
            get_decode_kernel,
            get_prefill_kernel,
        )
        from vllm.model_executor.layers.quantization.bitnet_dequant import (
            dequant_bias_triton,
        )

        # 1. Activation quantization: BF16 -> INT8 + scale (vLLM CUDA kernel)
        #    scale_a = absmax / 127, same convention as dequant's inv_scale_a
        x_flat = x.reshape(-1, K)
        x_int8, scale_a, _ = scaled_int8_quant(x_flat)

        # 2. Kernel dispatch
        if M == 1 and hasattr(layer, "weight_decode") and layer.weight_decode is not None:
            # Decode path (GEMV)
            kernel = get_decode_kernel(M, N, K)
            int32_out = torch.zeros(M, N, device=x.device, dtype=torch.int32)
            kernel(x_int8, layer.weight_decode, int32_out)
        else:
            # Prefill path (GEMM) — kernel compiled with padded M
            kernel, padded_m = get_prefill_kernel(M, N, K)
            if padded_m > M:
                x_padded = torch.zeros(
                    padded_m, K, device=x.device, dtype=torch.int8
                )
                x_padded[:M] = x_int8
                int32_out = kernel(x_padded, qweight)[:M]
            else:
                int32_out = kernel(x_int8, qweight)

        # 3. Fused dequant: offset correction + scale + bias in one Triton kernel
        #    scale_a from vLLM = absmax/127 = inv_scale_a needed by dequant
        scale_b = weight_scale.float().reshape(-1)  # [N]
        row_sum = x_int8.to(torch.int32).sum(dim=-1)  # [M] INT32 accumulate

        out_bf16 = dequant_bias_triton(
            int32_out, scale_a, scale_b,
            bias=bias,
            row_sum=row_sum,
            packing_offset=2,
        )

        return out_bf16.to(x.dtype).reshape(out_shape)

    def _apply_fallback(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        qweight: torch.Tensor,
        weight_scale: torch.Tensor,
        K: int,
        N: int,
        out_shape: tuple,
    ) -> torch.Tensor:
        """Torch fallback: unpack int2 -> dequant -> matmul."""
        # Unpack int2 -> int8 ternary {-1, 0, 1}
        p = qweight.data.to(torch.uint8)
        v0 = (p & 0x03).to(torch.int8) - 2
        v1 = ((p >> 2) & 0x03).to(torch.int8) - 2
        v2 = ((p >> 4) & 0x03).to(torch.int8) - 2
        v3 = ((p >> 6) & 0x03).to(torch.int8) - 2
        w_int8 = torch.stack([v0, v1, v2, v3], dim=-1).reshape(
            qweight.shape[0], K
        )  # [N, K]

        beta = weight_scale.float().reshape(-1)  # [N] per-channel scale
        w_float = w_int8.float() * beta[:, None]  # [N, K]

        x_flat = x.reshape(-1, K).float()
        out = torch.matmul(x_flat, w_float.t())  # [M, N]
        out = out.to(x.dtype).reshape(out_shape)

        if bias is not None:
            out = out + bias

        return out


class BitNetInt8EmbeddingMethod(QuantizeMethodBase):
    """INT8 per-channel quantized embedding for embed_tokens / lm_head.

    Saves ~2.5GB VRAM by storing vocab embeddings as INT8 + FP32 scale
    instead of BF16. Dequantized on-the-fly during lookup/matmul.
    """

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # INT8 weight: [V_partition, D]
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        # Need output_dim/input_dim for vLLM's weight sharding
        weight.output_dim = 0
        weight.input_dim = 1
        if weight_loader is not None:
            weight.weight_loader = weight_loader
        layer.register_parameter("weight", weight)

        # Per-channel scale: [V_partition, 1] float32
        weight_scale = Parameter(
            torch.ones(
                output_size_per_partition, 1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        weight_scale.output_dim = 0
        if weight_loader is not None:
            weight_scale.weight_loader = weight_loader
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
        )
        # Both embed_tokens and lm_head stay INT8 + scale.
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = Parameter(
            layer.weight_scale.data, requires_grad=False
        )
        # lm_head: store [K, N] contiguous for torch._int_mm.
        # Briefly needs 2x memory during .t().contiguous(), then original freed.
        # embed_tokens keeps row-major [N, K] for F.embedding row lookup.
        if isinstance(layer, ParallelLMHead):
            layer.weight_col_major = Parameter(
                layer.weight.data.t().contiguous(), requires_grad=False
            )
            layer.weight = None

    @torch.compiler.disable
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm._custom_ops import scaled_int8_quant
        from vllm.model_executor.layers.quantization.bitnet_dequant import (
            dequant_bias_triton,
        )

        # lm_head forward: INT8 activation × INT8 weight GEMM
        K = x.shape[-1]
        N = layer.weight_col_major.shape[1]
        out_shape = x.shape[:-1] + (N,)
        x_flat = x.reshape(-1, K)

        # Per-token activation quantization to INT8 (vLLM CUDA kernel)
        # scale_a = absmax / 127, [M, 1] float32
        x_int8, scale_a, _ = scaled_int8_quant(x_flat)

        # INT8 × INT8 GEMM → INT32
        # torch._int_mm requires M > 16, pad to 32 if needed (decode M=1)
        M = x_int8.shape[0]
        if M <= 16:
            x_padded = torch.zeros(32, K, dtype=torch.int8, device=x.device)
            x_padded[:M] = x_int8
            out_int32 = torch._int_mm(x_padded, layer.weight_col_major)[:M]
        else:
            out_int32 = torch._int_mm(x_int8, layer.weight_col_major)

        # Fused dequant: scale_a [M,1] * weight_scale [1,N] + bias (Triton)
        scale_b = layer.weight_scale.float().reshape(-1)  # [N]
        out_bf16 = dequant_bias_triton(
            out_int32, scale_a, scale_b,
            bias=bias,
        )

        return out_bf16.to(x.dtype).reshape(out_shape)

    @torch.compiler.disable
    def embedding(
        self, layer: nn.Module, input_: torch.Tensor
    ) -> torch.Tensor:
        # embed_tokens forward: lookup INT8 rows then per-row dequant.
        # Must NOT cast full weight to float32 (would allocate 4.7GB).
        # F.embedding works on int8 tensors — returns int8 rows.
        emb_int8 = F.embedding(input_, layer.weight)  # [..., K] int8
        scale_sel = F.embedding(input_, layer.weight_scale)  # [..., 1]
        return (emb_int8.float() * scale_sel).to(torch.bfloat16)
