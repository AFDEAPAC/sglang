"""gfx942 decode fusions that are bitwise the unfused ops they replace.

strided_per_token_quant_fp8: per-token fp8 quant of a [T, G, R] activation whose last dim is contiguous (the wo_a
einsum output), without the copy that flatten(1) makes; the arithmetic is _per_token_group_quant_8bit's.

silu_and_mul_clamp_per_token_quant_fp8: the clamped SwiGLU of _silu_and_mul_clamp_kernel, rounded to bf16 as that
kernel stores it, then per-token fp8 quant of the bf16 row, in one launch per row.

rmsnorm_with_per_token_quant: aiter's residual-free bf16 RMSNorm whose launch also writes the per-token fp8 quant of
its output (aiter rmsnorm_dual_ptq); apply_fp8_linear takes that pair in place of its own quant of the same tensor.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max


def _env_on(name: str) -> bool:
    return torch.version.hip is not None and os.environ.get(name, "0") == "1"


WOB_STRIDED_QUANT = _env_on("SGLANG_GFX942_WOB_STRIDED_QUANT")
SHARED_ACT_QUANT = _env_on("SGLANG_GFX942_SHARED_ACT_QUANT")
RMSNORM_ROW_STRIDED = _env_on("SGLANG_GFX942_RMSNORM_ROW_STRIDED")
LOWRATIO_INDEX_CACHE = _env_on("SGLANG_GFX942_LOWRATIO_INDEX_CACHE")
RMSNORM_PTQ = _env_on("SGLANG_GFX942_RMSNORM_PTQ")
_RMSNORM_PTQ_LOG = _env_on("SGLANG_GFX942_RMSNORM_PTQ_LOG")

logger = logging.getLogger(__name__)


def rowwise_fp8_linear_takes_prequant(linear) -> bool:
    """Whether linear's forward reaches apply_fp8_linear's rowwise (per-token x per-channel) fp8
    GEMM, which then accepts a pre-quantized (fp8, per-token scale) tuple as its input."""
    takes = getattr(linear, "_gfx942_takes_prequant", None)
    if takes is None:
        from sglang.srt.layers.quantization import w8a8_fp8

        scale = getattr(linear, "weight_scale", None)
        takes = bool(
            isinstance(getattr(linear, "quant_method", None), w8a8_fp8.W8A8Fp8LinearMethod)
            and w8a8_fp8._use_aiter
            and scale is not None
            and scale.numel() > 1
            and getattr(linear, "input_scale", None) is None
            and getattr(linear, "input_is_parallel", True)
        )
        linear._gfx942_takes_prequant = takes
    if not takes:
        return False
    from sglang.srt.runtime_context import get_forward

    return not get_forward().sp_active


@triton.jit
def _strided_per_token_quant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    stride_t,
    stride_g,
    R,
    N,
    eps,
    bit8_min,
    bit8_max,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    g = cols // R
    y = tl.load(x_ptr + t * stride_t + g * stride_g + (cols - g * R), mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, bit8_min, bit8_max).to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + t * N + cols, y_q, mask=mask)
    tl.store(s_ptr + t, y_s)


def strided_per_token_quant_fp8(x: torch.Tensor, eps: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor]:
    """[T, G, R] -> (fp8 [T, G * R], fp32 scale [T, 1]), bitwise per_token_group_quant_fp8(x.flatten(1), G * R)."""
    assert x.dim() == 3 and x.stride(-1) == 1, (x.shape, x.stride())
    T, G, R = x.shape
    N = G * R
    q = torch.empty((T, N), device=x.device, dtype=fp8_dtype)
    s = torch.empty((T, 1), device=x.device, dtype=torch.float32)
    if T:
        block = triton.next_power_of_2(N)
        _strided_per_token_quant_kernel[(T,)](
            x, q, s, x.stride(0), x.stride(1), R, N, eps,
            bit8_min=-fp8_max, bit8_max=fp8_max, BLOCK=block,
            num_warps=min(max(block // 256, 1), 8), num_stages=1,
        )
    return q, s


@triton.jit
def _silu_and_mul_clamp_quant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    inter_size,
    stride_xm,
    limit,
    eps,
    bit8_min,
    bit8_max,
    BLOCK_I: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_I)
    mask = offs < inter_size
    g = tl.load(x_ptr + pid_m * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + pid_m * stride_xm + inter_size + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    g = tl.minimum(g, limit)
    u = tl.minimum(tl.maximum(u, -limit), limit)
    y = g / (1.0 + tl.exp(-g)) * u
    y = y.to(tl.bfloat16).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, bit8_min, bit8_max).to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + pid_m * inter_size + offs, y_q, mask=mask)
    tl.store(s_ptr + pid_m, y_s)


def silu_and_mul_clamp_quant_supported(gate_up: torch.Tensor) -> bool:
    return (
        gate_up.dim() == 2
        and gate_up.dtype == torch.bfloat16
        and gate_up.stride(1) == 1
        and gate_up.shape[1] % 2 == 0
        and gate_up.shape[1] // 2 <= 1024
    )


def silu_and_mul_clamp_per_token_quant_fp8(
    gate_up: torch.Tensor, swiglu_limit: float, eps: float = 1e-10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """gate_up [M, 2 * I] (I <= 1024) -> (fp8 [M, I], fp32 scale [M, 1]), bitwise
    per_token_group_quant_fp8(silu_and_mul_clamp_triton(gate_up, swiglu_limit), I)."""
    M, N = gate_up.shape
    inter_size = N // 2
    q = torch.empty((M, inter_size), device=gate_up.device, dtype=fp8_dtype)
    s = torch.empty((M, 1), device=gate_up.device, dtype=torch.float32)
    if M:
        _silu_and_mul_clamp_quant_kernel[(M,)](
            gate_up, q, s, inter_size, gate_up.stride(0), float(swiglu_limit), eps,
            bit8_min=-fp8_max, bit8_max=fp8_max,
            BLOCK_I=1024 if inter_size >= 1024 else triton.next_power_of_2(inter_size),
            num_warps=4,
        )
    return q, s


def rmsnorm_ptq_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.stride(-1) == 1
        and x.shape[0] > 0
        and x.shape[1] <= 8192
        and x.shape[1] % 8 == 0
    )


def rmsnorm_with_per_token_quant(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """aiter rms_norm(x, weight, eps) for a 2-D bf16 x whose rows may be strided. The output carries, as _gfx942_ptq,
    the (fp8 [T, n], fp32 scale [T, 1]) that per_token_group_quant_fp8(output, n) would return, written by the same
    launch; it describes the output only while the output's _version is unchanged."""
    from aiter.ops.rmsnorm import rmsnorm_dual_ptq

    out = torch.empty_like(x)
    q = torch.empty(x.shape, device=x.device, dtype=fp8_dtype)
    s = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    rmsnorm_dual_ptq(out, q, s, x, weight, eps)
    out._gfx942_ptq = (q, s, out._version)
    return out


_ptq_events = {"hit": 0, "miss": 0}


def _log_ptq(kind: str, input_2d: torch.Tensor) -> None:
    _ptq_events[kind] += 1
    count = _ptq_events[kind]
    if count <= 48 or count & (count - 1) == 0:
        import traceback

        callers = " < ".join(f.name for f in traceback.extract_stack(limit=10)[-4::-1])
        logger.info("gfx942 rmsnorm ptq %s #%d M=%d K=%d via %s", kind, count, *input_2d.shape, callers)


def attached_per_token_quant(
    input: torch.Tensor, input_2d: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The (fp8, scale) that rmsnorm_with_per_token_quant attached to input, while it still describes input_2d."""
    ptq = getattr(input, "_gfx942_ptq", None)
    hit = (
        ptq is not None
        and ptq[2] == input._version
        and ptq[0].shape == input_2d.shape
        and input_2d.data_ptr() == input.data_ptr()
    )
    if _RMSNORM_PTQ_LOG and (hit or input_2d.shape[1] in (1280, 5120)):
        _log_ptq("hit" if hit else "miss", input_2d)
    return (ptq[0], ptq[1]) if hit else None
