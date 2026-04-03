# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.quark.schemes import QuarkMoEScheme
from sglang.srt.utils import get_bool_env_var, is_hip, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

__all__ = ["QuarkW4A8Int4Fp8MoE"]

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight


class QuarkW4A8Int4Fp8MoE(QuarkMoEScheme):
    """MoE scheme for pre-quantized W4A8 (INT4 weight + FP8 activation).

    Supports loading pre-quantized INT4 checkpoints with dual scales:
    - weight_scale: per-tensor FP8 scale
    - weight_scale_2: per-channel INT4 scale

    The dequantization formula is:
        original ~= int4_weight * weight_scale_2 * weight_scale
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def __init__(
        self,
        weight_configs: list[dict[str, Any]],
        input_config: dict[str, Any],
    ):
        self.int4_config = None
        self.fp8_config = None
        for wc in weight_configs:
            if isinstance(wc, dict):
                if wc.get("dtype") == "int4":
                    self.int4_config = wc
                elif wc.get("dtype") in (
                    "fp8_e4m3",
                    "fp8_e4m3fn",
                    "fp8_e4m3fnuz",
                ):
                    self.fp8_config = wc
        if self.int4_config is None or self.fp8_config is None:
            raise ValueError(
                f"W4A8 Int4Fp8 MoE requires both int4 and fp8 weight configs. "
                f"Got: {weight_configs}"
            )
        self.input_config = input_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 8,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 8,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs_tensor = dict(extra_weight_attrs)
        extra_weight_attrs_tensor["quant_method"] = (
            FusedMoeWeightScaleSupported.TENSOR.value
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs_tensor)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs_tensor)

        w13_weight_scale_2 = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale_2 = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)
        layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)
        extra_weight_attrs_channel = dict(extra_weight_attrs)
        extra_weight_attrs_channel["quant_method"] = (
            FusedMoeWeightScaleSupported.CHANNEL.value
        )
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs_channel)
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs_channel)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _use_aiter:
            raise NotImplementedError(
                "QuarkW4A8Int4Fp8MoE requires aiter "
                "(AMD GPU with SGLANG_USE_AITER=1)."
            )

        w13_int4_scale = layer.w13_weight_scale_2.data
        w2_int4_scale = layer.w2_weight_scale_2.data
        w13_fp8_scale = layer.w13_weight_scale.data
        w2_fp8_scale = layer.w2_weight_scale.data

        def _unpack_int32_reorder_to_int8(w_int32):
            orig_shape = w_int32.shape
            w_u8 = w_int32.view(torch.uint8)
            w_u8 = w_u8.view(*orig_shape[:-1], orig_shape[-1], 4)
            low = w_u8 & 0x0F
            high = (w_u8 >> 4) & 0x0F
            result = torch.empty(
                *orig_shape[:-1],
                orig_shape[-1] * 8,
                dtype=torch.int8,
                device=w_int32.device,
            )
            result[..., 0::8] = low[..., 0].to(torch.int8)
            result[..., 1::8] = low[..., 2].to(torch.int8)
            result[..., 2::8] = high[..., 0].to(torch.int8)
            result[..., 3::8] = high[..., 2].to(torch.int8)
            result[..., 4::8] = low[..., 1].to(torch.int8)
            result[..., 5::8] = low[..., 3].to(torch.int8)
            result[..., 6::8] = high[..., 1].to(torch.int8)
            result[..., 7::8] = high[..., 3].to(torch.int8)
            return result

        def _pack_shuffled_int8_to_flydsl_int4(x_i8):
            orig_shape = x_i8.shape
            flat = x_i8.contiguous().view(-1).to(torch.int16)
            u = (flat & 0xF).to(torch.uint8).view(-1, 8)
            out = torch.empty((u.shape[0], 4), device=u.device, dtype=torch.uint8)
            out[:, 0] = u[:, 0] | (u[:, 4] << 4)
            out[:, 1] = u[:, 1] | (u[:, 5] << 4)
            out[:, 2] = u[:, 2] | (u[:, 6] << 4)
            out[:, 3] = u[:, 3] | (u[:, 7] << 4)
            new_shape = list(orig_shape)
            new_shape[-1] = orig_shape[-1] // 2
            return out.view(-1).to(torch.int8).view(*new_shape)

        w13_unpacked = _unpack_int32_reorder_to_int8(layer.w13_weight.data)
        w2_unpacked = _unpack_int32_reorder_to_int8(layer.w2_weight.data)
        del layer.w13_weight, layer.w2_weight
        torch.cuda.empty_cache()

        w13_shuffled = shuffle_weight(w13_unpacked, (16, 16))
        del w13_unpacked
        torch.cuda.empty_cache()
        w2_shuffled = shuffle_weight(w2_unpacked, (16, 16))
        del w2_unpacked
        torch.cuda.empty_cache()

        layer.w13_weight = torch.nn.Parameter(
            _pack_shuffled_int8_to_flydsl_int4(w13_shuffled), requires_grad=False
        )
        del w13_shuffled
        torch.cuda.empty_cache()
        layer.w2_weight = torch.nn.Parameter(
            _pack_shuffled_int8_to_flydsl_int4(w2_shuffled), requires_grad=False
        )
        del w2_shuffled
        torch.cuda.empty_cache()

        shard_size = layer.intermediate_size_per_partition
        max_w13_scales = w13_fp8_scale.max(dim=1).values
        for expert_id in range(layer.num_experts):
            start = 0
            max_w13_scale_fp8 = max_w13_scales[expert_id]
            for shard_id in range(2):
                if w13_fp8_scale[expert_id][shard_id] != max_w13_scale_fp8:
                    int4_rescale = (
                        w13_fp8_scale[expert_id][shard_id] / max_w13_scale_fp8
                    )
                    w13_int4_scale[expert_id][
                        start : start + shard_size
                    ] *= int4_rescale
                start += shard_size

        for expert_id in range(layer.num_experts):
            w13_int4_scale[expert_id] *= max_w13_scales[expert_id]
            w2_int4_scale[expert_id] *= w2_fp8_scale[expert_id]

        layer.w13_int4_scale = torch.nn.Parameter(
            w13_int4_scale, requires_grad=False
        )
        layer.w2_int4_scale = torch.nn.Parameter(
            w2_int4_scale, requires_grad=False
        )
        layer.w13_fp8_scale = torch.nn.Parameter(
            max_w13_scales, requires_grad=False
        )
        layer.w2_fp8_scale = torch.nn.Parameter(
            w2_fp8_scale, requires_grad=False
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        topk_output = dispatch_output.topk_output
        moe_runner_config = self.moe_runner_config

        assert (
            not moe_runner_config.no_combine
        ), f"no_combine={moe_runner_config.no_combine} is not supported."

        output = fused_moe(
            dispatch_output.hidden_states,
            layer.w13_weight,
            layer.w2_weight,
            topk_output.topk_weights,
            topk_output.topk_ids,
            quant_type=QuantType.per_Token,
            w1_scale=layer.w13_int4_scale,
            w2_scale=layer.w2_int4_scale,
            activation=(
                ActivationType.Silu
                if moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            ),
        )

        return StandardCombineInput(hidden_states=output)
