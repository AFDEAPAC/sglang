"""HIP mHC sublayer boundary: hc_post + collapse + mixing statistics, with the
reduce + sinkhorn launched alone or hosted by the layer's next RMSNorm (HcCoefficients)."""

import ctypes
import logging
import os
import subprocess
import tempfile
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.layernorm.rmsnorm_fake_quant_amd_gfx95 import (
    Fp8GridActivation,
    Mxfp8Activation,
    rmsnorm_fake_quant_row,
    rmsnorm_row_chunk,
    row_major_2d,
)
from sglang.srt.environ import envs
from sglang.srt.utils.common import is_gfx942_supported, is_gfx95_supported, is_hip

logger = logging.getLogger(__name__)
_is_hip = is_hip()
# CDNA has no tf32: the mixing dot runs in ieee fp32 (the CUDA kernel uses tf32x3)
_HC_MIX_DOT_PRECISION = "ieee"

# a CTA owns every HC copy of its tile: a row's fp32 operation order depends on (H, HC), not on M
_HC_BOUNDARY_BLOCK_M = 16
_HC_BOUNDARY_BLOCK_K = 64
_HC_BOUNDARY_NUM_WARPS = 2
_HC_BOUNDARY_NUM_STAGES = 1
# From this many rows a 32-row tile is faster (MI355X). Both tiles compile with FMA contraction off: that keeps
# them bitwise each other and the gfx950 kernel (-ffp-contract=off).
_HC_BOUNDARY_WIDE_MIN_M = 96
_HC_BOUNDARY_WIDE = dict(
    BLOCK_M=32,
    num_warps=4,
    num_stages=2,
    kpack=2,
    matrix_instr_nonkdim=16,
    enable_fp_fusion=False,
)
# the reduce + sinkhorn row uses the norm kernel's warp count whether hosted there or launched alone
_HC_SINKHORN_NUM_WARPS = 4


def _hc_boundary_env_cfg(raw: str) -> Optional[Tuple[int, int, int]]:
    if raw in ("", "0") or is_gfx95_supported():
        return None
    block_m, block_k, num_warps = (int(v) for v in raw.split(","))
    return block_m, block_k, num_warps


# gfx942 tile overrides of the Triton partial kernel as "BLOCK_M,BLOCK_K,NUM_WARPS": DECODE for M <= 16 rows,
# PREFILL from SGLANG_GFX942_HC_BOUNDARY_PREFILL_MIN_M rows; BLOCK_K also sets the slice count the reduce sums
_HC_BOUNDARY_GFX942_DECODE = _hc_boundary_env_cfg(envs.SGLANG_GFX942_HC_BOUNDARY_DECODE.get())
_HC_BOUNDARY_GFX942_PREFILL = _hc_boundary_env_cfg(os.environ.get("SGLANG_GFX942_HC_BOUNDARY_PREFILL", ""))
_HC_BOUNDARY_GFX942_PREFILL_MIN_M = int(
    os.environ.get("SGLANG_GFX942_HC_BOUNDARY_PREFILL_MIN_M", "256")
)


def _hc_boundary_cfg(m: int) -> Tuple[int, int, int]:
    """(BLOCK_M, BLOCK_K, num_warps) of the partial kernel at m rows."""
    if _HC_BOUNDARY_GFX942_DECODE is not None and m <= 16:
        return _HC_BOUNDARY_GFX942_DECODE
    if (
        _HC_BOUNDARY_GFX942_PREFILL is not None
        and m >= _HC_BOUNDARY_GFX942_PREFILL_MIN_M
    ):
        return _HC_BOUNDARY_GFX942_PREFILL
    return _HC_BOUNDARY_BLOCK_M, _HC_BOUNDARY_BLOCK_K, _HC_BOUNDARY_NUM_WARPS


@triton.jit
def _hc_mix_reduce_sinkhorn_row(
    row,
    part_mix_ptr,
    part_sq_ptr,
    scratch_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    m,
    inv_k,
    rms_eps,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    SLICES_PAD: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
    PARTIALS_ROW_MAJOR: tl.constexpr,
):
    """_hc_mix_reduce_sinkhorn_kernel for one row with a tree fixed by NUM_SLICES; scratch_ptr
    ([m, 32] fp32) round-trips the reduced mixes so the sinkhorn starts from a plain layout."""
    j = tl.arange(0, HC)
    jj = j[:, None]
    kk = j[None, :]

    s_idx = tl.arange(0, SLICES_PAD)
    s_ok = s_idx < NUM_SLICES
    if PARTIALS_ROW_MAJOR:
        off = (row * NUM_SLICES + s_idx) * MIX
        sq_off = row * NUM_SLICES + s_idx
    else:
        off = (s_idx * m + row) * MIX
        sq_off = s_idx * m + row
    a_pre = tl.sum(
        tl.load(
            part_mix_ptr + off[:, None] + j[None, :],
            mask=s_ok[:, None],
            other=0.0,
        ),
        axis=0,
    )
    a_post = tl.sum(
        tl.load(
            part_mix_ptr + off[:, None] + HC + j[None, :],
            mask=s_ok[:, None],
            other=0.0,
        ),
        axis=0,
    )
    a_comb = tl.sum(
        tl.load(
            part_mix_ptr
            + off[:, None, None]
            + 2 * HC
            + jj[None, :, :] * HC
            + kk[None, :, :],
            mask=s_ok[:, None, None],
            other=0.0,
        ),
        axis=0,
    )
    sq = tl.sum(tl.load(part_sq_ptr + sq_off, mask=s_ok, other=0.0), axis=0)
    sp = scratch_ptr + row * 32
    tl.store(sp + j, a_pre)
    tl.store(sp + HC + j, a_post)
    tl.store(sp + 2 * HC + jj * HC + kk, a_comb)
    tl.debug_barrier()
    a_pre = tl.load(sp + j)
    a_post = tl.load(sp + HC + j)
    a_comb = tl.load(sp + 2 * HC + jj * HC + kk)
    rsqrt = 1.0 / tl.sqrt(sq * inv_k + rms_eps)

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    pre = tl.sigmoid(a_pre * rsqrt * s0 + tl.load(base_ptr + j)) + EPS
    tl.store(pre_ptr + row * HC + j, pre)
    post = 2.0 * tl.sigmoid(a_post * rsqrt * s1 + tl.load(base_ptr + HC + j))
    tl.store(post_ptr + row * HC + j, post)

    comb = a_comb * rsqrt * s2 + tl.load(base_ptr + 2 * HC + jj * HC + kk)
    comb = tl.exp(comb - tl.max(comb, axis=1)[:, None])
    comb = comb / tl.sum(comb, axis=1)[:, None] + EPS
    comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
    for _ in tl.static_range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + EPS)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
    tl.store(comb_ptr + row * HC * HC + jj * HC + kk, comb)


@triton.jit
def _hc_mix_reduce_sinkhorn_vec_kernel(
    part_mix_ptr,
    part_sq_ptr,
    scratch_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    m,
    inv_k,
    rms_eps,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    SLICES_PAD: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
    PARTIALS_ROW_MAJOR: tl.constexpr,
):
    """One program per row of _hc_mix_reduce_sinkhorn_row."""
    _hc_mix_reduce_sinkhorn_row(
        tl.program_id(0),
        part_mix_ptr,
        part_sq_ptr,
        scratch_ptr,
        scale_ptr,
        base_ptr,
        pre_ptr,
        post_ptr,
        comb_ptr,
        m,
        inv_k,
        rms_eps,
        MIX=MIX,
        HC=HC,
        NUM_SLICES=NUM_SLICES,
        SLICES_PAD=SLICES_PAD,
        ITERS=ITERS,
        EPS=EPS,
        PARTIALS_ROW_MAJOR=PARTIALS_ROW_MAJOR,
    )


@triton.jit
def _rmsnorm_sinkhorn_kernel(
    x_ptr,
    w_ptr,
    out_fq_ptr,
    out_norm_ptr,
    out_scale_ptr,
    K,
    stride_xm,
    stride_fm,
    stride_nm,
    stride_sm,
    eps,
    part_mix_ptr,
    part_sq_ptr,
    scratch_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    M,
    m,
    inv_k,
    rms_eps,
    WRITE_NORM: tl.constexpr,
    EMIT_FP8: tl.constexpr,
    FAKE_QUANT: tl.constexpr,
    CHUNK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    SLICES_PAD: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
    PARTIALS_ROW_MAJOR: tl.constexpr,
):
    """Grid (M + m,): programs [0, M) run rmsnorm_fake_quant_row on the norm's rows, programs
    [M, M + m) _hc_mix_reduce_sinkhorn_row on the pending boundary's rows; each row keeps its
    standalone kernel's arithmetic (same tiles, same warps)."""
    pid = tl.program_id(0)
    if pid < M:
        rmsnorm_fake_quant_row(
            pid.to(tl.int64),
            x_ptr,
            w_ptr,
            x_ptr,
            out_fq_ptr,
            out_norm_ptr,
            out_scale_ptr,
            K,
            stride_xm,
            0,
            stride_fm,
            stride_nm,
            stride_sm,
            eps,
            HAS_RESIDUAL=False,
            WRITE_NORM=WRITE_NORM,
            EMIT_FP8=EMIT_FP8,
            FAKE_QUANT=FAKE_QUANT,
            CHUNK=CHUNK,
            NUM_CHUNKS=NUM_CHUNKS,
        )
    else:
        _hc_mix_reduce_sinkhorn_row(
            pid - M,
            part_mix_ptr,
            part_sq_ptr,
            scratch_ptr,
            scale_ptr,
            base_ptr,
            pre_ptr,
            post_ptr,
            comb_ptr,
            m,
            inv_k,
            rms_eps,
            MIX=MIX,
            HC=HC,
            NUM_SLICES=NUM_SLICES,
            SLICES_PAD=SLICES_PAD,
            ITERS=ITERS,
            EPS=EPS,
            PARTIALS_ROW_MAJOR=PARTIALS_ROW_MAJOR,
        )


def hc_mix_reduce_sinkhorn_vec(
    part_mix: torch.Tensor,
    part_sq: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    k: int,
    rms_eps: float,
    mix: int,
    hc_mult: int,
    num_slices: int,
    sinkhorn_iters: int,
    hc_eps: float,
) -> None:
    """Reduce the [num_slices, m, mix] partials and run the sinkhorn into pre/post/comb."""
    m = part_sq.shape[1]
    if num_slices == 1 and _hc_sinkhorn_rows_gfx942(
        part_mix, part_sq, hc_scale, hc_base, pre, post, comb,
        k=k, rms_eps=rms_eps, mix=mix, hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, hc_eps=hc_eps,
    ):
        return
    # layout round trip for the reduce + sinkhorn row (see its docstring)
    scratch = torch.empty((m, 32), dtype=torch.float32, device=part_mix.device)
    _hc_mix_reduce_sinkhorn_vec_kernel[(m,)](
        part_mix,
        part_sq,
        scratch,
        hc_scale.float().contiguous(),
        hc_base.float().contiguous(),
        pre,
        post,
        comb,
        m,
        1.0 / k,
        rms_eps,
        MIX=mix,
        HC=hc_mult,
        NUM_SLICES=num_slices,
        SLICES_PAD=triton.next_power_of_2(num_slices),
        ITERS=sinkhorn_iters,
        EPS=hc_eps,
        PARTIALS_ROW_MAJOR=_partials_row_major(part_mix),
        num_warps=_HC_SINKHORN_NUM_WARPS,
    )


class HcCoefficients:
    """One boundary's mixing coefficients, held as split-K partials until rmsnorm_with_sinkhorn
    hosts their reduce + sinkhorn or the first access of pre / post / comb launches it."""

    def __init__(
        self,
        part_mix: torch.Tensor,
        part_sq: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        *,
        k: int,
        rms_eps: float,
        mix: int,
        hc_mult: int,
        num_slices: int,
        sinkhorn_iters: int,
        hc_eps: float,
    ):
        m = part_sq.shape[1]
        dev = part_mix.device
        self.part_mix = part_mix
        self.part_sq = part_sq
        self.hc_scale = hc_scale.float().contiguous()
        self.hc_base = hc_base.float().contiguous()
        self.k = k
        self.rms_eps = rms_eps
        self.mix = mix
        self.hc_mult = hc_mult
        self.num_slices = num_slices
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        self.num_rows = m
        self._pre = torch.empty(m, hc_mult, dtype=torch.float32, device=dev)
        self._post = torch.empty(m, hc_mult, dtype=torch.float32, device=dev)
        self._comb = torch.empty(m, hc_mult, hc_mult, dtype=torch.float32, device=dev)
        # layout round trip for the reduce + sinkhorn row (see its docstring)
        self.scratch = torch.empty((m, 32), dtype=torch.float32, device=dev)
        self.materialized = m == 0

    def materialize(self) -> None:
        """Run the reduce + sinkhorn alone unless a norm launch already hosted it."""
        if self.materialized:
            return
        self.materialized = True
        hc_mix_reduce_sinkhorn_vec(
            self.part_mix,
            self.part_sq,
            self.hc_scale,
            self.hc_base,
            self._pre,
            self._post,
            self._comb,
            k=self.k,
            rms_eps=self.rms_eps,
            mix=self.mix,
            hc_mult=self.hc_mult,
            num_slices=self.num_slices,
            sinkhorn_iters=self.sinkhorn_iters,
            hc_eps=self.hc_eps,
        )

    @property
    def pre(self) -> torch.Tensor:
        self.materialize()
        return self._pre

    @property
    def post(self) -> torch.Tensor:
        self.materialize()
        return self._post

    @property
    def comb(self) -> torch.Tensor:
        self.materialize()
        return self._comb

    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pre, self.post, self.comb


def rmsnorm_with_sinkhorn(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    coefficients: HcCoefficients,
    *,
    emit_fp8: bool = False,
    fake_quant: bool = True,
) -> Tuple[Union[Fp8GridActivation, Mxfp8Activation, None], torch.Tensor]:
    """rmsnorm_fake_quant_fp8(x, weight, eps) (the plain bf16 RMSNorm when fake_quant is
    False) with the pending reduce + sinkhorn of coefficients in the same launch; returns
    (fake_quant or None, norm)."""
    assert x.dim() == 2 and x.shape[-1] % 32 == 0, x.shape
    assert weight.dim() == 1 and weight.shape[0] == x.shape[-1], weight.shape
    assert weight.dtype == x.dtype, (weight.dtype, x.dtype)
    x = row_major_2d(x)
    weight = weight.contiguous()
    M, K = x.shape
    assert coefficients.num_rows == M, (
        f"coefficients hold {coefficients.num_rows} rows, the norm input {M}"
    )
    dev = x.device
    out_fq = (
        torch.empty(
            (M, K), dtype=torch.float8_e4m3fn if emit_fp8 else x.dtype, device=dev
        )
        if fake_quant
        else None
    )
    out_scale = (
        torch.empty((M, K // 32), dtype=torch.uint8, device=dev)
        if fake_quant and emit_fp8
        else None
    )
    out_norm = torch.empty((M, K), dtype=x.dtype, device=dev)
    if fake_quant:
        quant = (
            Mxfp8Activation(out_fq, out_scale)
            if emit_fp8
            else Fp8GridActivation(out_fq)
        )
    else:
        quant = None
    if M == 0:
        return quant, out_norm
    m = 0 if coefficients.materialized else M
    coefficients.materialized = True
    CHUNK = rmsnorm_row_chunk(K)
    _rmsnorm_sinkhorn_kernel[(M + m,)](
        x,
        weight,
        out_fq if out_fq is not None else x,
        out_norm,
        out_scale if out_scale is not None else x,
        K,
        x.stride(0),
        out_fq.stride(0) if out_fq is not None else 0,
        out_norm.stride(0),
        out_scale.stride(0) if out_scale is not None else 0,
        eps,
        coefficients.part_mix,
        coefficients.part_sq,
        coefficients.scratch,
        coefficients.hc_scale,
        coefficients.hc_base,
        coefficients._pre,
        coefficients._post,
        coefficients._comb,
        M,
        m,
        1.0 / coefficients.k,
        coefficients.rms_eps,
        WRITE_NORM=True,
        EMIT_FP8=emit_fp8,
        FAKE_QUANT=fake_quant,
        CHUNK=CHUNK,
        NUM_CHUNKS=triton.cdiv(K, CHUNK),
        MIX=coefficients.mix,
        HC=coefficients.hc_mult,
        NUM_SLICES=coefficients.num_slices,
        SLICES_PAD=triton.next_power_of_2(coefficients.num_slices),
        ITERS=coefficients.sinkhorn_iters,
        EPS=coefficients.hc_eps,
        PARTIALS_ROW_MAJOR=_partials_row_major(coefficients.part_mix),
        num_warps=_HC_SINKHORN_NUM_WARPS,
    )
    return quant, out_norm


@triton.jit
def _hc_boundary_partial_kernel(
    x_ptr,
    res_ptr,
    post_in_ptr,
    comb_in_ptr,
    pre_prev_ptr,
    w_ptr,
    res_out_ptr,
    y_ptr,
    part_mix_ptr,
    part_sq_ptr,
    M,
    H,
    x_stride_m,
    res_stride_m,
    w_stride_n,
    HC: tl.constexpr,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    HAS_POST: tl.constexpr,
    HAS_COMBINE: tl.constexpr,
    PARTIALS_ROW_MAJOR: tl.constexpr,
):
    """Grid (cdiv(M, BLOCK_M), H // BLOCK_K), one hidden slice of every copy per program. HAS_POST:
    res_out[k] = post[k]*x + sum_j comb[j,k]*res[j] in aiter::mhc_post's order, read back as bf16
    for the statistics; HAS_COMBINE: y = sum_k pre_prev[k] * copy_k in _hc_combine_kernel's order."""
    tl.static_assert(HC == 4)
    pid_m = tl.program_id(0)
    pid_t = tl.program_id(1)
    # int64: offs_m * (HC * H) passes 2**31 at ~105k rows of H=5120
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    mask_m = offs_m < M
    offs_h = pid_t * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, MIX_PAD)
    mask_n = offs_n < MIX
    m2 = mask_m[:, None]
    w_off = offs_n[None, :] * w_stride_n + offs_h[:, None]
    if HAS_POST:
        xa = tl.load(
            x_ptr + offs_m[:, None] * x_stride_m + offs_h[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        r_off = offs_m[:, None] * res_stride_m + offs_h[None, :]
        r0 = tl.load(res_ptr + r_off + 0 * H, mask=m2, other=0.0).to(tl.float32)
        r1 = tl.load(res_ptr + r_off + 1 * H, mask=m2, other=0.0).to(tl.float32)
        r2 = tl.load(res_ptr + r_off + 2 * H, mask=m2, other=0.0).to(tl.float32)
        r3 = tl.load(res_ptr + r_off + 3 * H, mask=m2, other=0.0).to(tl.float32)
        # one tile load per coefficient tensor; a one-hot sum extracts a column exactly
        cj = tl.arange(0, HC)
        post_tile = tl.load(
            post_in_ptr + offs_m[:, None] * HC + cj[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        cc = tl.arange(0, HC * HC)
        comb_tile = tl.load(
            comb_in_ptr + offs_m[:, None] * (HC * HC) + cc[None, :], mask=m2, other=0.0
        ).to(tl.float32)
    if HAS_COMBINE:
        pj = tl.arange(0, HC)
        pre_tile = tl.load(
            pre_prev_ptr + offs_m[:, None] * HC + pj[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        y = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, MIX_PAD], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k in tl.static_range(HC):
        if HAS_POST:
            pk = tl.sum(tl.where((cj == k)[None, :], post_tile, 0.0), axis=1)
            c0 = tl.sum(tl.where((cc == 0 * HC + k)[None, :], comb_tile, 0.0), axis=1)
            c1 = tl.sum(tl.where((cc == 1 * HC + k)[None, :], comb_tile, 0.0), axis=1)
            c2 = tl.sum(tl.where((cc == 2 * HC + k)[None, :], comb_tile, 0.0), axis=1)
            c3 = tl.sum(tl.where((cc == 3 * HC + k)[None, :], comb_tile, 0.0), axis=1)
            v = xa * pk[:, None]
            v += r0 * c0[:, None]
            v += r1 * c1[:, None]
            v += r2 * c2[:, None]
            v += r3 * c3[:, None]
            vb = v.to(res_out_ptr.dtype.element_ty)
            tl.store(
                res_out_ptr + offs_m[:, None] * (HC * H) + k * H + offs_h[None, :],
                vb,
                mask=m2,
            )
            xt = vb.to(tl.float32)
        else:
            xt = tl.load(
                x_ptr + offs_m[:, None] * x_stride_m + k * H + offs_h[None, :],
                mask=m2,
                other=0.0,
            ).to(tl.float32)
        if HAS_COMBINE:
            pp = tl.sum(tl.where((pj == k)[None, :], pre_tile, 0.0), axis=1)
            y += pp[:, None] * xt
        # loading each copy's tile where it is used keeps fewer live registers than prefetching all four
        w_tile = tl.load(w_ptr + w_off + k * H, mask=mask_n[None, :], other=0.0)
        acc += tl.dot(xt, w_tile, input_precision=DOT_PRECISION)
        sq += tl.sum(xt * xt, axis=1)
    if PARTIALS_ROW_MAJOR:
        mix_off = (offs_m * tl.num_programs(1) + pid_t) * MIX
        sq_off = offs_m * tl.num_programs(1) + pid_t
    else:
        mix_off = (pid_t * M + offs_m) * MIX
        sq_off = pid_t * M + offs_m
    tl.store(
        part_mix_ptr + mix_off[:, None] + offs_n[None, :],
        acc,
        mask=m2 & mask_n[None, :],
    )
    tl.store(part_sq_ptr + sq_off, sq, mask=mask_m)
    if HAS_COMBINE:
        tl.store(
            y_ptr + offs_m[:, None] * H + offs_h[None, :],
            y.to(y_ptr.dtype.element_ty),
            mask=m2,
        )


_HC_PARTIALS_ROW_MAJOR = envs.SGLANG_GFX942_HC_PARTIALS_ROW_MAJOR.get()


def _partials_row_major(part_mix: torch.Tensor) -> bool:
    """Whether [slices, m, mix] partials sit row-major in memory (see _hc_boundary_partials).
    One slice or one row is the same memory either way."""
    return part_mix.shape[0] > 1 and part_mix.shape[1] > 1 and part_mix.stride(0) == part_mix.shape[2]


# the gfx950 prefill kernel is bitwise the Triton kernel per row: the switch below is a speed choice
_HC_BOUNDARY_PREFILL_MIN_M = 1024


# the prefill kernel needs gfx950 (v_permlane*_swap, 16-byte LDS DMA)
_hc_boundary_prefill_available = is_gfx95_supported


@cache_once
def _hc_boundary_prefill_module():
    kernel = "mhc_boundary_hip::HcBoundaryPrefillKernel"
    return load_jit(
        "hc_boundary_prefill_hip",
        cuda_files=["deepseek_v4/mhc_boundary_gfx95.cuh"],
        cuda_wrappers=[
            ("post_combine", f"{kernel}<true>::run"),
            ("stats_only", f"{kernel}<false>::run"),
        ],
        # the hc_post chain must stay separate multiplies and adds, as in the Triton binary
        extra_cuda_cflags=["-ffp-contract=off"],
    )


def _hc_boundary_use_prefill(
    m: int, has_post: bool, has_combine: bool, x: Optional[torch.Tensor]
) -> bool:
    if m < _HC_BOUNDARY_PREFILL_MIN_M or not _hc_boundary_prefill_available():
        return False
    if (has_post, has_combine) not in ((True, True), (False, False)):
        return False
    # The LDS DMA moves 16-byte chunks: rows of x must stay 16-byte aligned.
    return x is None or x.stride(0) % 8 == 0


def _hc_boundary_partials(
    x: Optional[torch.Tensor],
    residual: torch.Tensor,
    post_in: Optional[torch.Tensor],
    comb_in: Optional[torch.Tensor],
    pre_prev: Optional[torch.Tensor],
    hc_fn: torch.Tensor,
    residual_out: Optional[torch.Tensor],
    y: Optional[torch.Tensor],
    *,
    hc_mult: int,
    prefill: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The boundary's first launch: writes residual_out / y when given
    and returns the [slices, M, MIX] mixing partials and [slices, M] row
    sums of squares. prefill forces a regime (tests); None selects by M."""
    m, _, h = residual.shape
    mix = hc_fn.shape[0]
    dev = residual.device
    has_post = x is not None
    has_combine = pre_prev is not None
    block_m, block_k, num_warps = _hc_boundary_cfg(m)
    num_slices = h // block_k
    if prefill is None:
        prefill = _hc_boundary_use_prefill(m, has_post, has_combine, x)
    # the gfx950 prefill kernel writes the contiguous layout
    row_major = _HC_PARTIALS_ROW_MAJOR and not prefill
    if row_major:
        part_mix = torch.empty((m, num_slices, mix), dtype=torch.float32, device=dev).transpose(0, 1)
        part_sq = torch.empty((m, num_slices), dtype=torch.float32, device=dev).transpose(0, 1)
    else:
        part_mix = torch.empty((num_slices, m, mix), dtype=torch.float32, device=dev)
        part_sq = torch.empty((num_slices, m), dtype=torch.float32, device=dev)
    if m == 0:
        return part_mix, part_sq
    placeholder = part_sq
    if prefill:
        mod = _hc_boundary_prefill_module()
        # CTAs per hidden slice: 8 waves each, one CTA per CU by LDS
        ctas = max(6, min(16, triton.cdiv(m, _HC_BOUNDARY_BLOCK_M) // 64))
        fn = mod.post_combine if has_post else mod.stats_only
        fn(
            x if has_post else placeholder,
            residual,
            post_in if has_post else placeholder,
            comb_in if has_post else placeholder,
            pre_prev if has_combine else placeholder,
            hc_fn,
            residual_out if has_post else placeholder,
            y if has_combine else placeholder,
            part_mix,
            part_sq,
            ctas,
        )
        return part_mix, part_sq
    mix_pad = max(16, triton.next_power_of_2(mix))
    # below 2x the minimum, a row count off the 32-row grid pads too much for the wide tile to win
    wide = m >= 2 * _HC_BOUNDARY_WIDE_MIN_M or (
        m >= _HC_BOUNDARY_WIDE_MIN_M and m % 32 == 0
    )
    if wide and envs.SGLANG_HIP_MHC_BOUNDARY_WIDE_TILE.get():
        launch = dict(_HC_BOUNDARY_WIDE)
    else:
        launch = dict(
            BLOCK_M=_HC_BOUNDARY_BLOCK_M,
            num_warps=_HC_BOUNDARY_NUM_WARPS,
            num_stages=_HC_BOUNDARY_NUM_STAGES,
            enable_fp_fusion=False,
        )
    if (_HC_BOUNDARY_GFX942_DECODE is not None and m <= 16) or (
        _HC_BOUNDARY_GFX942_PREFILL is not None
        and m >= _HC_BOUNDARY_GFX942_PREFILL_MIN_M
    ):
        # an explicit gfx942 tile outranks the wide-tile heuristic
        launch = dict(
            BLOCK_M=block_m,
            num_warps=num_warps,
            num_stages=_HC_BOUNDARY_NUM_STAGES,
            enable_fp_fusion=False,
        )
    block_m = launch["BLOCK_M"]
    grid_m = triton.cdiv(m, block_m)
    _hc_boundary_partial_kernel[(grid_m, num_slices)](
        x if has_post else residual,
        residual,
        post_in if has_post else placeholder,
        comb_in if has_post else placeholder,
        pre_prev if has_combine else placeholder,
        hc_fn,
        residual_out if has_post else placeholder,
        y if has_combine else placeholder,
        part_mix,
        part_sq,
        m,
        h,
        x.stride(0) if has_post else residual.stride(0),
        residual.stride(0),
        hc_fn.stride(0),
        HC=hc_mult,
        MIX=mix,
        MIX_PAD=mix_pad,
        BLOCK_K=block_k,
        DOT_PRECISION=_HC_MIX_DOT_PRECISION,
        HAS_POST=has_post,
        HAS_COMBINE=has_combine,
        PARTIALS_ROW_MAJOR=row_major,
        **launch,
    )
    return part_mix, part_sq


# gfx942 prefill boundaries: hc_post_rows_gfx942.hip writes residual_out, y and the row sums of squares with one
# wave per row (whole-row reads; the 64-column slices of the partial kernel read 64 B per row per instruction), then
# the mixing dots run as one GEMM against the weights split into bf16 hi + lo with fp32 output: one slice for the
# reduce + sinkhorn.
_HC_ROWS_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hc_post_rows_gfx942.hip")
_HC_ROWS_MIN_M = 1024
# Also run the Triton partials for the first N boundaries and log how the outputs compare.
_HC_ROWS_CHECK = int(os.environ.get("SGLANG_GFX942_HIP_HC_ROWS_CHECK", "0"))
_hc_rows_kernels = None
_hc_rows_checked = 0
_hc_sinkhorn_checked = 0


class _HcRowArgs(ctypes.Structure):
    _fields_ = (
        [(n, ctypes.c_void_p) for n in ("x", "res", "post_in", "comb_in", "pre_prev", "res_out", "y", "sq")]
        + [(n, ctypes.c_int64) for n in ("x_stride", "res_stride")]
        + [(n, ctypes.c_int32) for n in ("m", "h")]
    )


class _HcSinkArgs(ctypes.Structure):
    _fields_ = (
        [(n, ctypes.c_void_p) for n in ("mix", "sq", "scale", "base", "pre", "post", "comb")]
        + [(n, ctypes.c_int32) for n in ("m", "iters")]
        + [(n, ctypes.c_float) for n in ("inv_k", "rms_eps", "eps")]
    )


def _hc_rows_kernels_gfx942():
    """The (has_post, has_combine) row kernels, built with hipcc at first use; None when off or unavailable."""
    global _hc_rows_kernels
    if _hc_rows_kernels is None:
        _hc_rows_kernels = False
        if envs.SGLANG_GFX942_HIP_HC_BOUNDARY_ROWS.get() and is_gfx942_supported():
            from sglang.kernels.jit.utils.hipcc import (
                HipKernel,
                check_hip,
                hip_lib,
                hipcc_path,
            )

            co = os.path.join(tempfile.mkdtemp(prefix="sglang_hc_rows_"), "k.co")
            cmd = [hipcc_path(), "--genco", "--offload-arch=gfx942", "-O3", "-ffp-contract=off", "-o", co,
                   _HC_ROWS_SRC]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                mod = ctypes.c_void_p()
                check_hip(hip_lib().hipModuleLoad(ctypes.byref(mod), co.encode()), "load")
                _hc_rows_kernels = {
                    key: HipKernel(mod, f"hc_rows_gfx942_{name}")
                    for key, name in (
                        ((True, True), "post_comb"),
                        ((True, False), "post"),
                        ((False, True), "comb"),
                        ((False, False), "stats"),
                    )
                }
                _hc_rows_kernels["sinkhorn"] = HipKernel(mod, "hc_sinkhorn_rows_gfx942")
            except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
                logger.warning("gfx942 HIP HC boundary rows unavailable, keeping the Triton partials: %s", e)
    return _hc_rows_kernels or None


def _hc_sinkhorn_rows_gfx942(
    part_mix, part_sq, hc_scale, hc_base, pre, post, comb, *, k, rms_eps, mix, hc_mult, sinkhorn_iters, hc_eps
) -> bool:
    """The one-slice reduce + sinkhorn on hc_sinkhorn_rows_gfx942 (one thread per row; the Triton kernel runs a
    4-wave program per row: 6144 rows 112 us); False when it does not apply."""
    m = part_sq.shape[1]
    outs = (pre, post, comb)
    if (
        m < _HC_ROWS_MIN_M
        or part_mix.shape[0] != 1
        or hc_mult != 4
        or mix != 24
        or not envs.SGLANG_GFX942_HIP_HC_SINKHORN_ROWS.get()
        or part_mix.dtype != torch.float32
        or not (part_mix.is_contiguous() and part_sq.is_contiguous())
        or part_mix.data_ptr() % 16
        or any(t.dtype != torch.float32 or not t.is_contiguous() or t.data_ptr() % 16 for t in outs)
    ):
        return False
    kernels = _hc_rows_kernels_gfx942()
    if kernels is None:
        return False
    scale = hc_scale.float().contiguous()
    base = hc_base.float().contiguous()
    args = _HcSinkArgs(
        part_mix.data_ptr(), part_sq.data_ptr(), scale.data_ptr(), base.data_ptr(), pre.data_ptr(), post.data_ptr(),
        comb.data_ptr(), m, sinkhorn_iters, 1.0 / k, rms_eps, hc_eps,
    )
    kernels["sinkhorn"].launch(triton.cdiv(m, 256), args, torch.cuda.current_stream().cuda_stream, 256, 0)
    if _HC_ROWS_CHECK:
        _check_hc_sinkhorn(part_mix, part_sq, scale, base, outs, k, rms_eps, mix, hc_mult, sinkhorn_iters, hc_eps)
    return True


def _check_hc_sinkhorn(part_mix, part_sq, scale, base, outs, k, rms_eps, mix, hc_mult, sinkhorn_iters, hc_eps):
    global _hc_sinkhorn_checked
    if _hc_sinkhorn_checked >= _HC_ROWS_CHECK:
        return
    _hc_sinkhorn_checked += 1
    m = part_sq.shape[1]
    ref = tuple(torch.empty_like(t) for t in outs)
    _hc_mix_reduce_sinkhorn_vec_kernel[(m,)](
        part_mix, part_sq, torch.empty((m, 32), dtype=torch.float32, device=part_mix.device), scale, base, *ref,
        m, 1.0 / k, rms_eps, MIX=mix, HC=hc_mult, NUM_SLICES=1, SLICES_PAD=1, ITERS=sinkhorn_iters, EPS=hc_eps,
        PARTIALS_ROW_MAJOR=_partials_row_major(part_mix), num_warps=_HC_SINKHORN_NUM_WARPS,
    )
    logger.info(
        "gfx942 HIP HC sinkhorn check %d: rows=%d max|d pre| %.3g max|d post| %.3g max|d comb| %.3g",
        _hc_sinkhorn_checked,
        m,
        *(float((a - b).abs().max()) for a, b in zip(outs, ref)),
    )


def _hc_weights_hilo(hc_fn: torch.Tensor) -> torch.Tensor:
    """[64, k] bf16: hi rows, lo rows, zero rows of the fp32 mixing weights, cached on the weight tensor."""
    cached = getattr(hc_fn, "_gfx942_hilo", None)
    if cached is not None and cached[0] == hc_fn._version:
        return cached[1]
    hi = hc_fn.to(torch.bfloat16)
    pad = hi.new_zeros(64 - 2 * hc_fn.shape[0], hc_fn.shape[1])
    hilo = torch.cat([hi, (hc_fn - hi.float()).to(torch.bfloat16), pad], 0)
    hc_fn._gfx942_hilo = (hc_fn._version, hilo)
    return hilo


def _hc_boundary_rows_gfx942(x, residual, post_in, comb_in, pre_prev, hc_fn, residual_out, y):
    """([1, m, mix] partials, [1, m] sums of squares) from the row kernel + GEMM, or None where it does not apply."""
    m, hc_mult, h = residual.shape
    if (
        m < _HC_ROWS_MIN_M
        or hc_mult != 4
        or h % 512
        or residual.data_ptr() % 16
        or (x is not None and (x.stride(0) % 8 or x.data_ptr() % 16))
        # the hi + lo weight cache must not be created inside a graph capture
        or (torch.cuda.is_current_stream_capturing() and getattr(hc_fn, "_gfx942_hilo", None) is None)
    ):
        return None
    kernels = _hc_rows_kernels_gfx942()
    if kernels is None:
        return None
    if pre_prev is not None:
        pre_prev = pre_prev.float().contiguous()
    sq = torch.empty(m, dtype=torch.float32, device=residual.device)
    ptr = lambda t: 0 if t is None else t.data_ptr()  # noqa: E731
    args = _HcRowArgs(
        ptr(x), residual.data_ptr(), ptr(post_in), ptr(comb_in), ptr(pre_prev), ptr(residual_out), ptr(y),
        sq.data_ptr(), 0 if x is None else x.stride(0), residual.stride(0), m, h,
    )
    kernel = kernels[(x is not None, pre_prev is not None)]
    kernel.launch(triton.cdiv(m, 4), args, torch.cuda.current_stream().cuda_stream, 256, 0)
    mix = hc_fn.shape[0]
    copies = (residual if x is None else residual_out).view(m, hc_mult * h)
    hilo = _hc_weights_hilo(hc_fn)
    # hipBLASLt picks very different kernels across these shapes (k = 20480, MI308X): N padded to 64 is fastest below
    # 12288 rows (6144: 109 vs 241 us at N = 48), the transposed product above (16384: 371 vs 551 us)
    if m >= 12288:
        dots = torch.mm(hilo[: 2 * mix], copies.t(), out_dtype=torch.float32)
        mixes = (dots[:mix] + dots[mix : 2 * mix]).t().contiguous()
    else:
        dots = torch.mm(copies, hilo.t(), out_dtype=torch.float32)
        mixes = dots[:, :mix] + dots[:, mix : 2 * mix]
    part_mix, part_sq = mixes.unsqueeze(0), sq.unsqueeze(0)
    if _HC_ROWS_CHECK:
        _check_hc_rows(x, residual, post_in, comb_in, pre_prev, hc_fn, residual_out, y, part_mix, part_sq)
    return part_mix, part_sq


def _check_hc_rows(x, residual, post_in, comb_in, pre_prev, hc_fn, residual_out, y, part_mix, part_sq):
    global _hc_rows_checked
    if _hc_rows_checked >= _HC_ROWS_CHECK:
        return
    _hc_rows_checked += 1
    ref_out = None if residual_out is None else torch.empty_like(residual_out)
    ref_y = None if y is None else torch.empty_like(y)
    pm, ps = _hc_boundary_partials(
        x, residual, post_in, comb_in, pre_prev, hc_fn, ref_out, ref_y, hc_mult=residual.shape[1]
    )
    same = all(
        (a is None and b is None) or torch.equal(a, b) for a, b in ((residual_out, ref_out), (y, ref_y))
    )
    ref_mix, ref_sq = pm.sum(0), ps.sum(0)
    logger.info(
        "gfx942 HIP HC rows check %d: rows=%d post=%d combine=%d residual_out/y identical %s "
        "rel d(dots) %.3g rel d(sq) %.3g",
        _hc_rows_checked,
        residual.shape[0],
        x is not None,
        y is not None,
        same,
        float((part_mix[0] - ref_mix).abs().max() / ref_mix.abs().max()),
        float((part_sq[0] - ref_sq).abs().max() / ref_sq.abs().max()),
    )


def hc_boundary_fused_deferred(
    x: Optional[torch.Tensor],
    residual: torch.Tensor,
    post_in: Optional[torch.Tensor],
    comb_in: Optional[torch.Tensor],
    pre_prev: Optional[torch.Tensor],
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_eps: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], HcCoefficients]:
    """HIP mHC sublayer boundary with the reduce + sinkhorn left pending: residual_out =
    hc_post(x, residual, post_in, comb_in) when x is given, y = sum_k pre_prev[k] * copy_k
    when pre_prev is; returns (residual_out, y, coefficients), see HcCoefficients for how
    the last launch is hosted by the norm that follows the boundary."""
    assert _is_hip, "hc_boundary_fused_deferred launches HIP-only kernels"
    assert hc_mult == 4 and residual.dim() == 3 and residual.shape[1] == hc_mult
    assert residual.stride(2) == 1 and residual.stride(1) == residual.shape[2]
    assert hc_fn.dtype == torch.float32 and hc_fn.stride(1) == 1
    m, _, h = residual.shape
    k = hc_mult * h
    mix = hc_fn.shape[0]
    assert mix == (2 + hc_mult) * hc_mult and hc_fn.shape[1] == k
    assert h % _hc_boundary_cfg(m)[1] == 0, h
    dev = residual.device
    has_post = x is not None
    has_combine = pre_prev is not None
    if has_post:
        assert post_in is not None and comb_in is not None
        assert x.shape == (m, h) and x.stride(1) == 1
        post_in = post_in.contiguous().float()
        comb_in = comb_in.contiguous().float()
        residual_out = torch.empty_like(residual)
    else:
        residual_out = None
    if has_combine:
        pre_prev = pre_prev.contiguous()
        y = torch.empty((m, h), dtype=residual.dtype, device=dev)
    else:
        y = None
    rows = _hc_boundary_rows_gfx942(
        x, residual, post_in, comb_in, pre_prev, hc_fn, residual_out, y
    )
    if rows is not None:
        part_mix, part_sq = rows
    else:
        part_mix, part_sq = _hc_boundary_partials(
            x,
            residual,
            post_in,
            comb_in,
            pre_prev,
            hc_fn,
            residual_out,
            y,
            hc_mult=hc_mult,
        )
    coefficients = HcCoefficients(
        part_mix,
        part_sq,
        hc_scale,
        hc_base,
        k=k,
        rms_eps=rms_eps,
        mix=mix,
        hc_mult=hc_mult,
        num_slices=part_sq.shape[0],
        sinkhorn_iters=sinkhorn_iters,
        hc_eps=hc_eps,
    )
    return residual_out, y, coefficients


@triton.jit
def _rmsnorm_ptq_sinkhorn_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    q_ptr,
    s_ptr,
    K,
    stride_xm,
    eps,
    fp8_max,
    part_mix_ptr,
    part_sq_ptr,
    scratch_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    M,
    m,
    inv_k,
    rms_eps,
    BLOCK: tl.constexpr,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    SLICES_PAD: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
    PARTIALS_ROW_MAJOR: tl.constexpr,
):
    """Grid (M + m,): programs [0, M) run the residual-free bf16 RMSNorm of a row and the per-token fp8 quant
    of its bf16 output (gfx942_decode_fuse.rmsnorm_with_per_token_quant), programs [M, M + m)
    _hc_mix_reduce_sinkhorn_row on the pending boundary's rows with its standalone arithmetic."""
    pid = tl.program_id(0)
    if pid < M:
        row = pid.to(tl.int64)
        offs = tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
        rcp = tl.rsqrt(tl.sum(x * x, axis=0) / K + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x * rcp * w).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * K + offs, y, mask=mask)
        yf = y.to(tl.float32)
        y_s = tl.maximum(tl.max(tl.abs(yf), axis=0), 1e-10) / fp8_max
        y_s_inv = 1.0 / y_s
        y_q = tl.clamp(yf * y_s_inv, -fp8_max, fp8_max).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + row * K + offs, y_q, mask=mask)
        tl.store(s_ptr + row, y_s)
    else:
        _hc_mix_reduce_sinkhorn_row(
            pid - M,
            part_mix_ptr,
            part_sq_ptr,
            scratch_ptr,
            scale_ptr,
            base_ptr,
            pre_ptr,
            post_ptr,
            comb_ptr,
            m,
            inv_k,
            rms_eps,
            MIX=MIX,
            HC=HC,
            NUM_SLICES=NUM_SLICES,
            SLICES_PAD=SLICES_PAD,
            ITERS=ITERS,
            EPS=EPS,
            PARTIALS_ROW_MAJOR=PARTIALS_ROW_MAJOR,
        )


def rmsnorm_ptq_with_sinkhorn(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    coefficients: HcCoefficients,
) -> torch.Tensor:
    """gfx942_decode_fuse.rmsnorm_with_per_token_quant(x, weight, eps) with the pending reduce + sinkhorn of
    coefficients in the same launch: the bf16 norm, carrying its (fp8 [M, K], fp32 scale [M, 1]) as _gfx942_ptq."""
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max

    assert x.dim() == 2 and x.stride(1) == 1, (x.shape, x.stride())
    M, K = x.shape
    assert coefficients.num_rows == M, (
        f"coefficients hold {coefficients.num_rows} rows, the norm input {M}"
    )
    dev = x.device
    out = torch.empty((M, K), dtype=x.dtype, device=dev)
    q = torch.empty((M, K), dtype=fp8_dtype, device=dev)
    s = torch.empty((M, 1), dtype=torch.float32, device=dev)
    m = 0 if coefficients.materialized else M
    coefficients.materialized = True
    if M:
        _rmsnorm_ptq_sinkhorn_kernel[(M + m,)](
            x,
            weight,
            out,
            q,
            s,
            K,
            x.stride(0),
            eps,
            fp8_max,
            coefficients.part_mix,
            coefficients.part_sq,
            coefficients.scratch,
            coefficients.hc_scale,
            coefficients.hc_base,
            coefficients._pre,
            coefficients._post,
            coefficients._comb,
            M,
            m,
            1.0 / coefficients.k,
            coefficients.rms_eps,
            BLOCK=triton.next_power_of_2(K),
            MIX=coefficients.mix,
            HC=coefficients.hc_mult,
            NUM_SLICES=coefficients.num_slices,
            SLICES_PAD=triton.next_power_of_2(coefficients.num_slices),
            ITERS=coefficients.sinkhorn_iters,
            EPS=coefficients.hc_eps,
            PARTIALS_ROW_MAJOR=_partials_row_major(coefficients.part_mix),
            num_warps=_HC_SINKHORN_NUM_WARPS,
        )
    from sglang.kernels.ops.quantization.gfx942_decode_fuse import _ptq_version

    out._gfx942_ptq = (q, s, _ptq_version(out))
    return out
