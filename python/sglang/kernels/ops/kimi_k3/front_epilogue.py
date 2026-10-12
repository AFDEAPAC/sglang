"""K3_FRONT_EPILOGUE: shared SiTU + dense routed latent from the fused MoE front, one launch."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args
from sglang.kernels.ops.kimi_k3.activation import _fast_math_flags, _make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module

ENABLED = os.environ.get("SGLANG_K3_FRONT_EPILOGUE", "0") not in ("", "0", "false", "False")
_announced = False


@cache_once
def _jit_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        _make_name("situ_front_epilogue"),
        *args,
        cuda_files=["kimi_k3/situ_front_epilogue.cuh"],
        cuda_wrappers=[("run", f"SituFrontEpilogueKernel<{args}>::run")],
        extra_cuda_cflags=_fast_math_flags(),
    )


def situ_front_epilogue(fused, gate_up, routed, act_fn):
    """Return (shared_act [T, D], routed_dense [T, R]) or (None, routed) if not applicable.

    ``gate_up`` and ``routed`` must be column slices of the 2D bf16 ``fused`` tensor.
    """
    global _announced
    beta = getattr(act_fn, "beta", None)
    linear_beta = getattr(act_fn, "linear_beta", None)
    if (
        beta is None
        or fused.dim() != 2
        or fused.dtype != torch.bfloat16
        or fused.stride(1) != 1
        or gate_up.data_ptr() != fused.data_ptr()
    ):
        return None, routed
    elem = fused.element_size()
    routed_col = (routed.data_ptr() - fused.data_ptr()) // elem
    num_tokens = fused.shape[0]
    d = gate_up.shape[1] // 2
    r = routed.shape[1]
    vec = 16 // elem
    if d % vec or r % vec or routed_col % vec or fused.stride(0) % vec:
        return None, routed
    shared_act = fused.new_empty((num_tokens, d))
    routed_dense = fused.new_empty((num_tokens, r))
    has_linear_beta = linear_beta is not None
    _jit_module(fused.dtype).run(
        fused,
        shared_act,
        routed_dense,
        int(routed_col),
        float(beta),
        float(linear_beta) if has_linear_beta else 0.0,
        has_linear_beta,
    )
    if not _announced:
        _announced = True
        print(f"[k3-front-epilogue] active D={d} R={r} routed_col={routed_col}", flush=True)
    return shared_act, routed_dense
