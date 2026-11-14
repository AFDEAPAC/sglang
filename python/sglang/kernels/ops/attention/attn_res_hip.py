"""Triton attention-residual aggregation for Kimi-K3 on ROCm.

The HIP counterpart of attn_res.py: same aggregation point (score the bank rows
against the current prefix, softmax, weighted sum, output RMSNorm), one launch,
but built for a GPU with no TMA and no tcgen05. See _agg_kernel for why the
shape differs so much from the SM100 kernel's.
"""

from __future__ import annotations

from functools import cache
from typing import Optional

import torch
import triton
import triton.language as tl

# _agg_kernel keeps a next_pow2(nvb) x next_pow2(H) fp32 tile in registers, and
# that is the whole basis of its speed. Past this budget it spills and loses to
# the 2-kernel Triton pipeline it replaces: measured at T=4 on MI355X,
# next_pow2(nvb)=8 is 2.0x faster, 16 is 1.2x, 32 is 0.2x. K3 sits right at the
# limit with H=7168 (one masked tile of 8192) and nvb <= 8.
MAX_REGISTER_TILE: int = 8 * 8192

# The fp8 output reproduces aiter.per_token_quant_hip on gfx942 (e4m3fnuz) bit
# for bit: scale = amax * (1/240), inverse by v_rcp_f32, clamp by v_med3_f32,
# convert by v_cvt_pk_fp8_f32 (aiter's fp32_to_fp8_scaled_x2).
_FP8_FNUZ_MAX = 240.0


@cache
def supports_attn_res_hip(hidden_size: int, nvb: int) -> bool:
    """Whether this shape fits the register budget. Callers must additionally
    check that they are on ROCm; this is only the shape constraint."""
    return _tile_size(hidden_size, nvb) <= MAX_REGISTER_TILE


def _tile_size(hidden_size: int, nvb: int) -> int:
    return triton.next_power_of_2(max(nvb, 1)) * triton.next_power_of_2(hidden_size)


@triton.jit
def _agg_kernel(
    prefix_ptr,  # [T, H]
    addend_ptr,  # [T, H]; the pending residual, or prefix_ptr when not HAS_ADD
    prefix_out_ptr,  # [T, H]; materialized prefix, written when HAS_ADD
    bank_ptr,  # [T, NB, H]
    cw_ptr,  # [H] fp32; score_norm weight * score_proj weight
    ow_ptr,  # [H]; out RMSNorm weight, unread when not APPLY_OUT_NORM
    out_ptr,  # [T, H]
    out_q_ptr,  # [T, H] fp8, written when QUANT_OUT
    out_s_ptr,  # [T] fp32 per-token scale, written when QUANT_OUT
    score_eps,
    out_eps,
    stride_pm,
    stride_am,
    stride_om,
    stride_bm,
    stride_bb,
    stride_o,
    stride_q,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NVB: tl.constexpr,
    R_PAD: tl.constexpr,
    HAS_ADD: tl.constexpr,
    WRITE_BANK: tl.constexpr,
    APPLY_OUT_NORM: tl.constexpr,
    QUANT_OUT: tl.constexpr = False,
    FP8_MAX: tl.constexpr = 240.0,
    INV_FP8_MAX: tl.constexpr = 1.0,
):
    """One CTA per token: score the NVB+1 rows, softmax, mix, apply the output
    RMSNorm, all in one launch.

    T is the decode batch size, so this runs a handful of CTAs on a 256-CU GPU
    and what binds is the load latency and bandwidth of a *single* CU. That is
    what dictates the shape, and why it is not a port of the SM100 kernel. That
    one's online softmax reads each row once but chains one block-wide reduction
    per row; here each link in that chain costs a full HBM round-trip —
    measured 1.1us/row, dead linear in NVB, because there is no TMA pipeline to
    hide it behind. Scoring the rows as one [R_PAD, BLOCK_H] tile instead puts
    every row's load in flight at once, and keeping that tile in registers lets
    the mix reuse it rather than re-reading the bank, which at one active CU is
    the difference between ~9us and ~12.5us at NVB=8.

    Holding the tile is why NVB is a constexpr and why MAX_REGISTER_TILE caps
    the shape: the register budget is what this trades away.

    The prefix row streams through anyway, so the pending residual add and the
    bank snapshot ride along for free — no other program reads bank row NVB, so
    those stores need no synchronization.

    Taking the global max before exponentiating makes this bit-comparable to
    the 2-kernel pipeline rather than to the SM100 kernel.
    """
    t = tl.program_id(0).to(tl.int64)  # t * stride_bm overflows int32
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    # The prefix row is score row NVB, and the only row that needs writing back.
    row = tl.load(prefix_ptr + t * stride_pm + offs, mask=mask, other=0.0)
    if HAS_ADD:
        # Round to the storage dtype before scoring: downstream readers see
        # these bits, so the score has to as well.
        row = (
            row.to(tl.float32)
            + tl.load(addend_ptr + t * stride_am + offs, mask=mask, other=0.0).to(
                tl.float32
            )
        ).to(prefix_out_ptr.dtype.element_ty)
        tl.store(prefix_out_ptr + t * stride_om + offs, row, mask=mask)
    if WRITE_BANK:
        tl.store(bank_ptr + t * stride_bm + NVB * stride_bb + offs, row, mask=mask)
    pv = row.to(tl.float32)

    cw = tl.load(cw_ptr + offs, mask=mask, other=0.0)
    p_score = tl.sum(pv * cw) / tl.sqrt(tl.sum(pv * pv) / H + score_eps)

    # The whole bank, in registers: every row's load is in flight at once, and
    # the mix below reuses it instead of going back to HBM.
    offs_r = tl.arange(0, R_PAD)
    mask_r = offs_r < NVB
    tile = tl.load(
        bank_ptr + t * stride_bm + offs_r[:, None] * stride_bb + offs[None, :],
        mask=mask_r[:, None] & mask[None, :],
        other=0.0,
    ).to(tl.float32)
    b_score = tl.sum(tile * cw[None, :], axis=1) / tl.sqrt(
        tl.sum(tile * tile, axis=1) / H + score_eps
    )

    m = tl.maximum(tl.max(tl.where(mask_r, b_score, -float("inf"))), p_score)
    b_w = tl.where(mask_r, tl.exp(b_score - m), 0.0)
    p_w = tl.exp(p_score - m)
    inv = 1.0 / (tl.sum(b_w) + p_w)

    # Explicit fma: the compiler contracts this mul+add in some variants of
    # the kernel and not in others, which would change the output bits.
    acc = tl.fma(pv, p_w * inv, tl.sum(b_w[:, None] * inv * tile, axis=0))

    if APPLY_OUT_NORM:
        scale = 1.0 / tl.sqrt(tl.sum(acc * acc) / H + out_eps)
        ow = tl.load(ow_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc = acc * scale * ow
    y = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + t * stride_o + offs, y, mask=mask)
    if QUANT_OUT:
        # Quantize the stored bf16 values, as a separate quant kernel would.
        yf = y.to(tl.float32)
        q_scale = tl.max(tl.where(mask, tl.abs(yf), 0.0)) * INV_FP8_MAX
        inv = tl.inline_asm_elementwise(
            "v_rcp_f32 $0, $1", "=v,v", [q_scale], dtype=tl.float32, is_pure=True, pack=1
        )
        lo = tl.full([BLOCK_H], -FP8_MAX, tl.float32)
        hi = tl.full([BLOCK_H], FP8_MAX, tl.float32)
        qf = tl.inline_asm_elementwise(
            "v_med3_f32 $0, $1, $2, $3",
            "=v,v,v,v",
            [yf * inv, lo, hi],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        packed = tl.inline_asm_elementwise(
            "v_cvt_pk_fp8_f32 $0, $1, $1", "=v,v", [qf], dtype=tl.int32, is_pure=True, pack=1
        )
        q = packed.to(tl.uint8).to(out_q_ptr.dtype.element_ty, bitcast=True)
        # Cap the store at 8 contiguous bytes: a 16-byte fp8 store makes the
        # compiler give every thread 16 elements instead of 8, which reorders
        # the reductions above and changes the bf16 output.
        offs_q = tl.max_contiguous(tl.multiple_of(offs, 8), 8)
        tl.store(out_q_ptr + t * stride_q + offs_q, q, mask=mask)
        tl.store(out_s_ptr + t, q_scale)


def attn_res_hip(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: Optional[torch.Tensor],
    out: torch.Tensor,
    nvb: int,
    score_eps: float,
    out_eps: float,
    *,
    addend: Optional[torch.Tensor] = None,
    prefix_out: Optional[torch.Tensor] = None,
    write_prefix: bool = False,
    out_q: Optional[torch.Tensor] = None,
    out_s: Optional[torch.Tensor] = None,
) -> None:
    """Single-kernel attention-residual aggregation for ROCm.

    Restrictions: nvb >= 1, and the shape must pass supports_attn_res_hip().

    Parameters
    ----------
    prefix_sum : [T, H] bf16 — the running prefix, or its first term if addend
                 is given
    bank       : [T, NB, H] bf16 (rows 0..nvb-1 are aggregated)
    cw         : [H] fp32 — precomputed score_norm weight * proj weight
    ow         : [H] output RMSNorm weight, or None to return the pre-norm
                 softmax mixture (the aggregate-stream value)
    out        : [T, H] bf16 output buffer
    nvb        : number of valid bank rows (>= 1)
    score_eps, out_eps : RMSNorm epsilons; unlike the SM100 kernel these need
                 not be equal
    addend     : fold a pending residual add in, so the aggregated prefix is
                 prefix_sum + addend; requires prefix_out
    prefix_out : [T, H] bf16 buffer receiving that materialized prefix
    write_prefix : also snapshot the prefix row into bank[:, nvb, :] (bit-exact
                 copy, fused into the score pass which already has the row in
                 registers); requires NB > nvb
    out_q, out_s : [T, H] float8_e4m3fnuz and [T, 1] fp32 buffers receiving
                 aiter.per_token_quant_hip(out) bit for bit (gfx942 only)
    """
    T, H = prefix_sum.shape
    assert nvb >= 1, "nvb == 0 has nothing to aggregate; the caller must handle it"
    assert supports_attn_res_hip(H, nvb), (
        f"attn_res_hip: register tile {_tile_size(H, nvb)} exceeds "
        f"{MAX_REGISTER_TILE} (H={H}, nvb={nvb})"
    )
    has_add = addend is not None
    assert not has_add or prefix_out is not None, "addend requires prefix_out"
    quant_out = out_q is not None
    if quant_out:
        assert out_s is not None and out_q.dtype == torch.float8_e4m3fnuz
        assert out_q.shape == out.shape and out_q.stride(1) == 1 and out_s.is_contiguous()

    # Triton needs a real pointer for every argument; the flags decide whether
    # these are ever dereferenced.
    addend_arg = addend if has_add else prefix_sum
    prefix_out_arg = prefix_out if has_add else prefix_sum
    ow_arg = ow if ow is not None else cw

    import os as _os_agg
    _nw = int(_os_agg.environ.get("SGLANG_K3_AGG_NUM_WARPS", "4"))
    if not getattr(_os_agg, "_k3_agg_nw_announced", False):
        print(f"[k3-agg-warps] n={_nw}", flush=True)
        _os_agg._k3_agg_nw_announced = True
    _agg_kernel[(T,)](
        prefix_sum,
        addend_arg,
        prefix_out_arg,
        bank,
        cw,
        ow_arg,
        out,
        out_q if quant_out else out,
        out_s if quant_out else cw,
        score_eps,
        out_eps,
        prefix_sum.stride(0),
        addend_arg.stride(0),
        prefix_out_arg.stride(0),
        bank.stride(0),
        bank.stride(1),
        out.stride(0),
        out_q.stride(0) if quant_out else 0,
        H=H,
        BLOCK_H=triton.next_power_of_2(H),
        NVB=nvb,
        R_PAD=triton.next_power_of_2(nvb),
        HAS_ADD=has_add,
        WRITE_BANK=write_prefix,
        APPLY_OUT_NORM=ow is not None,
        QUANT_OUT=quant_out,
        FP8_MAX=_FP8_FNUZ_MAX,
        INV_FP8_MAX=1.0 / _FP8_FNUZ_MAX,
        num_warps=_nw,
    )
