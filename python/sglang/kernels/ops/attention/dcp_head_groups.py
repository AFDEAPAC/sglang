"""Glue kernels for the gfx942 DCP MLA decode on 64-head groups.

gfx942's only MLA decode kernel that returns the LSE takes 64 heads per call, so
``AiterAttnBackend._forward_decode_dcp_head_groups`` runs the gathered heads as
zero-padded groups. These two kernels replace the torch ops around those calls:
``dcp_head_group_q`` builds the fp8 ``[n_groups, bs, group, D]`` input from a bf16
``[bs, heads, D]`` q, and ``dcp_head_group_finish`` gathers the group outputs and
LSEs back to ``[bs, heads, V]`` and ``[bs, heads]``, zeroing the output and setting
the LSE to -inf for rows with no KV on this rank. Both produce the same bytes as the
torch ops they replace.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.kvcache.mla_buffer import TORCH_FP8_OVERFLOW_AT


@triton.jit
def _head_group_q_kernel(
    q_ptr,
    out_ptr,
    bs,
    q_stride_b,
    q_stride_h,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OVERFLOW_NAN_AT: tl.constexpr,
):
    # One program per output row; rows are ordered (group, batch, head in group).
    row = tl.program_id(0)
    hh = row % G
    gb = row // G
    b = gb % bs
    h = (gb // bs) * G + hh
    offs = tl.arange(0, BLOCK_D)
    in_row = offs < D
    x = tl.load(
        q_ptr + b * q_stride_b + h * q_stride_h + offs,
        mask=in_row & (h < H),
        other=0.0,
    )
    x = tl.where(tl.abs(x) >= OVERFLOW_NAN_AT, float("nan"), x)
    tl.store(out_ptr + row * D + offs, x.to(out_ptr.dtype.element_ty), mask=in_row)


@triton.jit
def _head_group_finish_kernel(
    o_ptr,
    lse0_ptr,
    lse1_ptr,
    empty_ptr,
    out_ptr,
    lse_out_ptr,
    bs,
    H: tl.constexpr,
    G: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # One program per (batch, head) of the output.
    pid = tl.program_id(0)
    h = pid % H
    b = pid // H
    g = h // G
    hh = h - g * G
    empty = tl.load(empty_ptr + b)
    offs = tl.arange(0, BLOCK_V)
    in_row = offs < V
    x = tl.load(o_ptr + ((g * bs + b) * G + hh) * V + offs, mask=in_row)
    x = tl.where(empty, tl.zeros_like(x), x)
    tl.store(out_ptr + pid * V + offs, x, mask=in_row)
    l0 = tl.load(lse0_ptr + b * G + hh, mask=g == 0, other=0.0)
    l1 = tl.load(lse1_ptr + b * G + hh, mask=g == 1, other=0.0)
    lse = tl.where(g == 0, l0, l1)
    lse = tl.where(empty, float("-inf"), lse)
    tl.store(lse_out_ptr + pid, lse)


def dcp_head_group_q(
    q: torch.Tensor, n_groups: int, group: int, fp8_dtype: torch.dtype
) -> torch.Tensor:
    """bf16 ``[bs, heads, D]`` -> fp8 ``[n_groups, bs, group, D]``, padded heads zero.
    ``fp8_dtype`` must be in ``TORCH_FP8_OVERFLOW_AT``."""
    bs, num_heads, head_dim = q.shape
    assert q.stride(-1) == 1 and num_heads <= n_groups * group
    out = torch.empty(
        (n_groups, bs, group, head_dim), dtype=fp8_dtype, device=q.device
    )
    _head_group_q_kernel[(n_groups * bs * group,)](
        q,
        out,
        bs,
        q.stride(0),
        q.stride(1),
        H=num_heads,
        G=group,
        D=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        OVERFLOW_NAN_AT=TORCH_FP8_OVERFLOW_AT[fp8_dtype],
    )
    return out


def dcp_head_group_finish(
    o: torch.Tensor, lses, empty: torch.Tensor, num_heads: int
):
    """Group outputs ``[n_groups, bs, group, V]`` and per-group LSEs ``[bs, group]``
    (at most two groups) -> ``([bs, heads, V], [bs, heads])``; rows with ``empty``
    set get a zero output and a -inf LSE."""
    n_groups, bs, group, v_dim = o.shape
    assert 1 <= len(lses) <= 2 and len(lses) == n_groups and o.is_contiguous()
    assert all(l.is_contiguous() and l.shape == (bs, group) for l in lses)
    out = torch.empty((bs, num_heads, v_dim), dtype=o.dtype, device=o.device)
    lse = torch.empty((bs, num_heads), dtype=lses[0].dtype, device=o.device)
    _head_group_finish_kernel[(bs * num_heads,)](
        o,
        lses[0],
        lses[-1],
        empty,
        out,
        lse,
        bs,
        H=num_heads,
        G=group,
        V=v_dim,
        BLOCK_V=triton.next_power_of_2(v_dim),
    )
    return out, lse
