# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz, scaled_fp8_quant
from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz
from sglang.srt.layers.quantization.utils import all_close_1d, per_tensor_dequantize
from sglang.srt.utils import (
    get_bool_env_var,
    is_gfx95_supported,
    is_hip,
    set_weight_attrs,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.quark.quark import QuarkConfig

logger = logging.getLogger(__name__)

_is_shuffle_moe_mxfp4 = is_gfx95_supported()

__all__ = ["QuarkMoEMethod", "QuarkW4A4MXFp4MoEMethod"]

_is_fp8_fnuz = is_fp8_fnuz()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    from sglang.srt.layers.moe.rocm_moe_utils import rocm_fused_experts_tkw1

OCP_MX_BLOCK_SIZE = 32

if TYPE_CHECKING:
    from sglang.srt.layers.quantization import QuarkConfig


class QuarkMoEMethod(FusedMoEMethodBase):

    def __init__(self, quant_config: QuarkConfig):
        self.quant_config = quant_config

    @staticmethod
    def get_moe_method(
        quant_config: QuarkConfig,  # type: ignore # noqa E501 # noqa F821
        module: torch.nn.Module,
        layer_name: str,
    ) -> "QuarkMoEMethod":
        layer_quant_config = quant_config._find_matched_config(layer_name, module)

        if layer_quant_config.get("output_tensors") or layer_quant_config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with "
                "output_tensors and bias "
                "quantized are not supported"
            )
        weight_config = layer_quant_config.get("weight")
        input_config = layer_quant_config.get("input_tensors")

        # Handle W4A8 dual-quantization: weight_config is a list with
        # both int4 (per-channel) and fp8 (per-tensor) configs
        if isinstance(weight_config, list):
            has_int4 = any(
                isinstance(wc, dict) and wc.get("dtype") == "int4"
                for wc in weight_config
            )
            has_fp8 = any(
                isinstance(wc, dict)
                and wc.get("dtype") in ("fp8_e4m3", "fp8_e4m3fn", "fp8_e4m3fnuz")
                for wc in weight_config
            )
            if has_int4 and has_fp8:
                return QuarkW4A8Int4Fp8MoEMethod(weight_config, input_config)

            # Fallback: try each config individually
            for wc in weight_config:
                if quant_config._is_mx_fp4(wc, input_config):
                    return QuarkW4A4MXFp4MoEMethod(wc, input_config)
                elif quant_config._is_fp8_w8a8(wc, input_config):
                    return QuarkW8A8FP8MoEMethod(wc, input_config)

            raise RuntimeError(
                f"Unsupported FusedMoe scheme. "
                f"Weight config: {weight_config}, Input config: {input_config}"
            )

        if quant_config._is_mx_fp4(weight_config, input_config):
            return QuarkW4A4MXFp4MoEMethod(weight_config, input_config)
        elif quant_config._is_fp8_w8a8(weight_config, input_config):
            return QuarkW8A8FP8MoEMethod(weight_config, input_config)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme. "
                f"Weight config: {weight_config}, Input config: {input_config}"
            )


class QuarkW4A4MXFp4MoEMethod(QuarkMoEMethod):

    def __init__(self, weight_config: dict[str, Any], input_config: dict[str, Any]):
        self.weight_quant = weight_config
        self.input_quant = input_config

        weight_qscheme = self.weight_quant.get("qscheme")
        input_qscheme = self.input_quant.get("qscheme")
        if not (weight_qscheme == "per_group" and input_qscheme == "per_group"):
            raise ValueError(
                "For MX(FP4) Fused MoE layers, only per-group scales "
                "for weights and activations are supported. Found "
                f"{weight_qscheme}, {input_qscheme}"
            )  # noqa E501

        self.static_input_scales = not self.input_quant.get("is_dynamic")
        self.with_bias = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )

        params_dtype = torch.uint8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)

        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)

        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        float_dtype = torch.get_default_dtype()

        # Pre-shuffle weight scales
        s0, s1, _ = layer.w13_weight_scale.shape
        w13_weight_scale = layer.w13_weight_scale.view(s0 * s1, -1)
        w13_weight_scale = e8m0_shuffle(w13_weight_scale)
        # layer.w13_weight_scale = torch.nn.Parameter(w13_weight_scale, requires_grad=False)
        layer.w13_weight_scale.data = w13_weight_scale.view(s0, s1, -1)

        s0, s1, _ = layer.w2_weight_scale.shape
        w2_weight_scale = layer.w2_weight_scale.view(s0 * s1, -1)
        w2_weight_scale = e8m0_shuffle(w2_weight_scale)
        # layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)
        layer.w2_weight_scale.data = w2_weight_scale.view(s0, s1, -1)

        # Pre-shuffle weight
        if _is_shuffle_moe_mxfp4:
            layer.w13_weight.data = shuffle_weight(
                layer.w13_weight.contiguous(), (16, 16)
            )
            layer.w2_weight.data = shuffle_weight(
                layer.w2_weight.contiguous(), (16, 16)
            )
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        moe_runner_config = self.moe_runner_config
        topk_weights, topk_ids, _ = topk_output
        if _is_hip:
            topk_weights = topk_weights.to(
                torch.float32
            )  # aiter's moe_sorting requires topk_weights to be FP32

        if hasattr(torch, "float4_e2m1fn_x2"):
            w13_weight = layer.w13_weight.view(torch.float4_e2m1fn_x2)
            w2_weight = layer.w2_weight.view(torch.float4_e2m1fn_x2)
        else:
            w13_weight = layer.w13_weight
            w2_weight = layer.w2_weight

        if hasattr(layer.w13_weight, "is_shuffled"):
            w13_weight.is_shuffled = True
            w2_weight.is_shuffled = True

        output = fused_moe(
            x,
            w13_weight,
            w2_weight,
            topk_weights,
            topk_ids,
            quant_type=QuantType.per_1x32,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            activation=(
                ActivationType.Silu
                if moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            ),
            doweight_stage1=False,
            expert_mask=layer.expert_mask_gpu,
        )
        return StandardCombineInput(hidden_states=output)


class QuarkW8A8FP8MoEMethod(QuarkMoEMethod):

    def __init__(self, weight_config: dict[str, Any], input_config: dict[str, Any]):
        self.is_static_input_scheme: bool = False
        self.input_qscheme = None

        if input_config is not None:
            self.is_static_input_scheme = not input_config.get("is_dynamic")
            self.input_qscheme = input_config.get("qscheme")

        self.input_per_token = (
            not self.is_static_input_scheme and self.input_qscheme == "per_channel"
        )
        self.weight_qscheme = weight_config.get("qscheme")
        self.is_weight_per_channel = self.weight_qscheme == "per_channel"
        self.out_dtype = torch.get_default_dtype()

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        # per-tensor quantization
        if self.weight_qscheme == "per_tensor":
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            weight_quant_method = FusedMoeWeightScaleSupported.TENSOR.value
        elif self.weight_qscheme == "per_channel":
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            weight_quant_method = FusedMoeWeightScaleSupported.CHANNEL.value
        else:
            raise ValueError(
                f"Unsupported weight quantization strategy: {self.weight_qscheme}."
            )

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update({"quant_method": weight_quant_method})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.is_static_input_scheme:
            assert (
                self.input_qscheme == "per_tensor"
            ), "Only per-tensor quantization is supported for static input scales"
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fp8 moe kernels require a single activation scale.
        # We take the max of all the scales in case they differ.
        if self.is_static_input_scheme:
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                layer.w2_input_scale
            ):
                logger.warning(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer."
                )
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )

        if _is_fp8_fnuz:
            # Normalize the weights and scales
            w13_weight, w13_weight_scale, w13_input_scale = (
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                )
            )
            w2_weight, w2_weight_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
            )
            # Reset the parameter
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(
                w13_weight_scale, requires_grad=False
            )
            if w13_input_scale is not None:
                layer.w13_input_scale = torch.nn.Parameter(
                    w13_input_scale, requires_grad=False
                )
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_weight_scale, requires_grad=False
            )
            if w2_input_scale is not None:
                layer.w2_input_scale = torch.nn.Parameter(
                    w2_input_scale, requires_grad=False
                )
        if self.weight_qscheme == "per_tensor":
            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.num_local_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id],
                    )
                    (
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        _,
                    ) = scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])

                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )
        elif self.weight_qscheme == "per_channel":
            layer.w13_weight_scale = torch.nn.Parameter(
                layer.w13_weight_scale.unsqueeze(-1), requires_grad=False
            )
            layer.w2_weight_scale = torch.nn.Parameter(
                layer.w2_weight_scale.unsqueeze(-1), requires_grad=False
            )
        else:
            raise ValueError(
                f"Unsupported weight quantization strategy: {self.weight_qscheme}."
            )

        if _use_aiter and self.is_weight_per_channel:
            with torch.no_grad():
                layer.w13_weight = torch.nn.Parameter(
                    shuffle_weight(layer.w13_weight.data, (16, 16)),
                    requires_grad=False,
                )
                layer.w13_weight.is_shuffled = True
                torch.cuda.empty_cache()
                layer.w2_weight = torch.nn.Parameter(
                    shuffle_weight(layer.w2_weight.data, (16, 16)),
                    requires_grad=False,
                )
                layer.w2_weight.is_shuffled = True
                torch.cuda.empty_cache()

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        if (
            _use_aiter
            and self.is_weight_per_channel
            and moe_runner_config.apply_router_weight_on_input
        ):
            topk_weights, topk_ids, _ = topk_output
            output = rocm_fused_experts_tkw1(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=moe_runner_config.activation,
                apply_router_weight_on_input=moe_runner_config.apply_router_weight_on_input,
                use_fp8_w8a8=True,
                per_channel_quant=self.is_weight_per_channel,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )
            return StandardCombineInput(hidden_states=output)
        elif _use_aiter and self.is_weight_per_channel:
            topk_weights, topk_ids, _ = topk_output
            activation = (
                ActivationType.Silu
                if moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            )
            output = fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights.to(torch.float32),
                topk_ids.to(torch.int32),
                activation=activation,
                quant_type=QuantType.per_Token,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            )
            return StandardCombineInput(hidden_states=output)
        else:
            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                use_fp8_w8a8=True,
                per_channel_quant=self.is_weight_per_channel,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a13_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )
            return self.runner.run(dispatch_output, quant_info)


class QuarkW4A8Int4Fp8MoEMethod(QuarkMoEMethod):
    """MoE method for pre-quantized W4A8 (INT4 weight + FP8 activation).

    Supports loading pre-quantized INT4 checkpoints with dual scales:
    - weight_scale: per-tensor FP8 scale
    - weight_scale_2: per-channel INT4 scale

    The dequantization formula is:
        original ≈ int4_weight * weight_scale_2 * weight_scale
    """

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
                elif wc.get("dtype") in ("fp8_e4m3", "fp8_e4m3fn", "fp8_e4m3fnuz"):
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
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        # INT4 packed weights (8 int4 values per int32)
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

        # Per-tensor FP8 scales (mapped from checkpoint's weight_scale)
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

        # Per-channel INT4 scales (mapped from checkpoint's weight_scale_2)
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

        # No static input scales for dynamic activation quantization
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _use_aiter:
            raise NotImplementedError(
                "QuarkW4A8Int4Fp8MoEMethod requires aiter (AMD GPU with SGLANG_USE_AITER=1)."
            )

        # Alias for clarity
        w13_int4_scale = layer.w13_weight_scale_2.data
        w2_int4_scale = layer.w2_weight_scale_2.data
        w13_fp8_scale = layer.w13_weight_scale.data
        w2_fp8_scale = layer.w2_weight_scale.data

        # === Weight format conversion ===
        # The FlyDSL int4 kernel expects:
        #   1. Unpacked int8 (1 int4 per byte) → preshuffle → repack to FlyDSL format
        # Our checkpoint has int32 packed (aiter reorder format).
        # Flow: unpack int32 → int8 → shuffle_weight → _pack_to_flydsl_int4

        def _unpack_int32_reorder_to_int8(w_int32):
            """Unpack aiter-reorder int32 to sequential int8 (one int4 per byte)."""
            orig_shape = w_int32.shape  # [..., K/8]
            # View int32 as uint8 bytes: [..., K/8*4] = [..., K/2]
            w_u8 = w_int32.view(torch.uint8)
            w_u8 = w_u8.view(*orig_shape[:-1], orig_shape[-1], 4)  # [..., K/8, 4]
            low = w_u8 & 0x0F
            high = (w_u8 >> 4) & 0x0F
            # Aiter reorder: byte[0]=(v2<<4|v0), byte[1]=(v6<<4|v4),
            #                 byte[2]=(v3<<4|v1), byte[3]=(v7<<4|v5)
            result = torch.empty(
                *orig_shape[:-1], orig_shape[-1] * 8,
                dtype=torch.int8, device=w_int32.device
            )
            result[..., 0::8] = low[..., 0].to(torch.int8)   # v0
            result[..., 1::8] = low[..., 2].to(torch.int8)   # v1
            result[..., 2::8] = high[..., 0].to(torch.int8)  # v2
            result[..., 3::8] = high[..., 2].to(torch.int8)  # v3
            result[..., 4::8] = low[..., 1].to(torch.int8)   # v4
            result[..., 5::8] = low[..., 3].to(torch.int8)   # v5
            result[..., 6::8] = high[..., 1].to(torch.int8)  # v6
            result[..., 7::8] = high[..., 3].to(torch.int8)  # v7
            return result

        def _pack_shuffled_int8_to_flydsl_int4(x_i8):
            """Pack preshuffled int8 → FlyDSL packed int4 bytes.
            [v0..v7] → b0=(v4<<4)|v0, b1=(v5<<4)|v1, b2=(v6<<4)|v2, b3=(v7<<4)|v3.
            """
            orig_shape = x_i8.shape
            flat = x_i8.contiguous().view(-1).to(torch.int16)
            u = (flat & 0xF).to(torch.uint8).view(-1, 8)
            out = torch.empty((u.shape[0], 4), device=u.device, dtype=torch.uint8)
            out[:, 0] = u[:, 0] | (u[:, 4] << 4)
            out[:, 1] = u[:, 1] | (u[:, 5] << 4)
            out[:, 2] = u[:, 2] | (u[:, 6] << 4)
            out[:, 3] = u[:, 3] | (u[:, 7] << 4)
            # Reshape: last dim halved (2 int4 per byte)
            new_shape = list(orig_shape)
            new_shape[-1] = orig_shape[-1] // 2
            return out.view(-1).to(torch.int8).view(*new_shape)

        # Step 1: Unpack int32 → int8 (one int4 per byte, sequential column order)
        w13_unpacked = _unpack_int32_reorder_to_int8(layer.w13_weight.data)
        w2_unpacked = _unpack_int32_reorder_to_int8(layer.w2_weight.data)
        del layer.w13_weight, layer.w2_weight
        torch.cuda.empty_cache()

        # Step 2: Preshuffle the unpacked int8 data
        w13_shuffled = shuffle_weight(w13_unpacked, (16, 16))
        del w13_unpacked
        torch.cuda.empty_cache()
        w2_shuffled = shuffle_weight(w2_unpacked, (16, 16))
        del w2_unpacked
        torch.cuda.empty_cache()

        # Step 3: Pack shuffled int8 → FlyDSL packed int4 (2 per byte, int8 dtype)
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

        # Combine w1/w3 per-tensor FP8 scales into a single per-expert scale.
        # For W4A8, the full dequant is: original = int4_w * weight_scale_2 * weight_scale
        # We merge both scales into one per-channel scale for the kernel.
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

        # Merge: combined_scale = int4_per_channel_scale * fp8_per_tensor_scale
        for expert_id in range(layer.num_experts):
            w13_int4_scale[expert_id] *= max_w13_scales[expert_id]
            w2_int4_scale[expert_id] *= w2_fp8_scale[expert_id]

        # Store the processed scales as new parameters for apply()
        layer.w13_int4_scale = torch.nn.Parameter(w13_int4_scale, requires_grad=False)
        layer.w2_int4_scale = torch.nn.Parameter(w2_int4_scale, requires_grad=False)
        layer.w13_fp8_scale = torch.nn.Parameter(max_w13_scales, requires_grad=False)
        layer.w2_fp8_scale = torch.nn.Parameter(w2_fp8_scale, requires_grad=False)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
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

        # W4A8: int4 weights + fp8 dynamic activation quantization.
        # QuantType.per_Token triggers fp8 activation quant in fused_moe,
        # and routes to FlyDSL's "int4_fp8" kernel.
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
