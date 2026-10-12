"""HIP adapter for aiter #4712 fused KDA decode (conv + recurrence + gated RMSNorm)."""

from __future__ import annotations

import torch

from sglang.srt.utils import is_hip

_HEADS = 12
_DIM = 128
# seg is the per-tensor width of gate / out_gate (H*K); mixed_qkv and the conv pool are
# 3x that because q, k and v are packed on the same row. Mixing the two up is what
# silently rejected every batch on 2026-09-15 (see knowledge/known_bad.md).
_SEG = _HEADS * _DIM
_CHANNELS = 3 * _SEG


def enabled() -> bool:
    from sglang.srt.environ import envs

    return is_hip() and envs.SGLANG_AITER_FUSED_KDA_DECODE.get()


def covered(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    onorm_g: torch.Tensor,
) -> bool:
    if not enabled() or mixed_qkv.ndim != 2:
        return False
    bsz = mixed_qkv.shape[0]
    return (
        bsz > 0
        and mixed_qkv.shape == (bsz, _CHANNELS)
        and mixed_qkv.dtype == torch.bfloat16
        and mixed_qkv.stride(-1) == 1
        and a.ndim == 2
        and a.shape == (bsz, _SEG)
        and a.dtype == torch.bfloat16
        and a.stride(-1) == 1
        and b.ndim == 2
        and b.shape == (bsz, _HEADS)
        and b.dtype == torch.bfloat16
        and b.stride(-1) == 1
        and onorm_g.ndim == 2
        and onorm_g.shape == (bsz, _SEG)
        and onorm_g.dtype == torch.bfloat16
        and onorm_g.stride(-1) == 1
        and conv_states.ndim == 3
        and conv_states.shape[-1] == _CHANNELS
        and conv_states.shape[-2] == 3
        and conv_states.dtype == torch.bfloat16
        and conv_states.stride(-1) == 1
        and ssm_states.ndim == 4
        and ssm_states.shape[1:] == (_HEADS, _DIM, _DIM)
        and ssm_states.dtype == torch.float32
        and cache_indices.shape == (bsz,)
        and cache_indices.dtype == torch.int32
    )


def run(
    *,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_states: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    onorm_g: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    num_v_heads: int,
    head_v_dim: int,
) -> torch.Tensor:
    from aiter.ops.triton.gated_delta_net.fused_kda_decode import fused_kda_decode

    bsz = mixed_qkv.shape[0]
    gate = a.view(1, bsz, _HEADS, _DIM)
    beta3 = b.view(1, bsz, _HEADS) if b.ndim == 2 else b[0]
    # conv_state MUST be the live cache, passed as a strided VIEW.
    # The kernel rotates the conv history in place
    # (fused_conv_recurrent_norm.py:92-98 stores back through conv_state_ptr), and the
    # wrapper reads stride_cs_slot/dim/pos straight off the tensor, so a non-contiguous
    # view is by design. An earlier .contiguous() here allocated a throwaway copy, the
    # rotation landed in it, the real cache never advanced, and GSM8K fell to 0.005.
    out = fused_kda_decode(
        mixed_qkv=mixed_qkv,
        conv_state=conv_states.transpose(-1, -2),
        conv_weight=conv_weight,
        gate=gate,
        beta=beta3,
        out_gate=onorm_g,
        A_log=A_log.reshape(-1),
        dt_bias=dt_bias.reshape(-1),
        ssm_state=ssm_states,
        ssm_state_indices=cache_indices,
        cu_seqlens=query_start_loc,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        head_dim=head_v_dim,
        num_local_heads=num_v_heads,
        lower_bound=lower_bound,
    )
    return out.view(1, bsz, num_v_heads, head_v_dim)
