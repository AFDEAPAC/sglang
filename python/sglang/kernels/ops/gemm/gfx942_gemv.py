"""gfx942 skinny GEMVs for decode row counts.

fp8_gemv_rowwise: the per-token x per-channel fp8 linear of apply_fp8_linear's rowwise path for M <= 16 rows, on
aiter's (16, 16) pre-shuffled weight layout or a row-major one; fp32 split-K partials summed in a fixed order by a
second launch that also applies both scales.

bf16_gemv: x[M, K] @ w[N, K].T for a handful of rows (the lm_head), fp32 accumulation.

Neither is bitwise the GEMM it replaces: the fp32 accumulation order differs from CK's / hipBLASLt's.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl
from sglang.srt.environ import envs


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


FP8_GEMV_MAX_M = envs.SGLANG_GFX942_FP8_GEMV_MAX_M.get() if torch.version.hip is not None else 0
# which weight layouts the GEMV takes: all, rowmajor (the unshuffled K=288 / 576 linears) or shuffled
_FP8_GEMV_LAYOUT = envs.SGLANG_GFX942_FP8_GEMV_LAYOUT.get()
# k-blocks of 32 per tile and the program count the split-K aims for
_FP8_GEMV_KB = _env_int("SGLANG_GFX942_FP8_GEMV_KB", 4)
_FP8_GEMV_PROGRAMS = _env_int("SGLANG_GFX942_FP8_GEMV_PROGRAMS", 1024)
_FP8_GEMV_WARPS = _env_int("SGLANG_GFX942_FP8_GEMV_WARPS", 2)

LM_HEAD_GEMV_MAX_M = _env_int("SGLANG_GFX942_LM_HEAD_GEMV_MAX_M", 0) if torch.version.hip is not None else 0
_LM_HEAD_BLOCK_N = _env_int("SGLANG_GFX942_LM_HEAD_BLOCK_N", 64)
_LM_HEAD_BLOCK_K = _env_int("SGLANG_GFX942_LM_HEAD_BLOCK_K", 256)
_LM_HEAD_WARPS = _env_int("SGLANG_GFX942_LM_HEAD_WARPS", 8)
_LM_HEAD_STAGES = _env_int("SGLANG_GFX942_LM_HEAD_STAGES", 2)


@triton.jit
def _fp8_gemv_partial_kernel(
    x_ptr,
    w_ptr,
    xs_ptr,
    ws_ptr,
    out_ptr,
    part_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_om,
    SHUFFLED: tl.constexpr,
    KB: tl.constexpr,
    TILES_PER_SPLIT: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """Grid (N // 16, SPLIT): 16 output channels times one K range. SHUFFLED reads aiter's (16, 16) fp8 layout, in
    which channel block nb, k-block kb (32 wide), half s and channel r hold bytes (nb * K/32 + kb) * 512 + s * 256
    + r * 16 + (k % 16); otherwise w is row-major [N, K]."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    KT: tl.constexpr = KB * 32
    offs_m = tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    offs_k = tl.arange(0, KT)
    mask_m = offs_m < M
    n0 = pid_n * 16
    w_base = w_ptr + n0.to(tl.int64) * K
    if SHUFFLED:
        # memory order: row q = (k-block * 2 + half) * 16 + channel holds 16 consecutive k
        q = tl.arange(0, KT)
        w_tile = q[:, None] * 16 + offs_n[None, :]
        q_kb = q // 32
    else:
        w_tile = offs_n[None, :].to(tl.int64) * K + offs_k[:, None]
    acc = tl.zeros([16, 16], dtype=tl.float32)
    for t in tl.static_range(TILES_PER_SPLIT):
        k0 = (pid_s * TILES_PER_SPLIT + t) * KT
        mask_k = k0 + offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        if SHUFFLED:
            w_raw = tl.load(
                w_base + k0 * 16 + w_tile, mask=(k0 + q_kb * 32 < K)[:, None], other=0.0
            )
            # [k-block * 2 + half, channel, k % 16] -> [k, channel]
            w = tl.reshape(
                tl.permute(tl.reshape(w_raw, (KB * 2, 16, 16)), (0, 2, 1)), (KT, 16)
            )
        else:
            w = tl.load(w_base + k0 + w_tile, mask=mask_k[:, None], other=0.0)
        acc += tl.dot(x, w)
    n = n0 + offs_n
    if SPLIT == 1:
        xs = tl.load(xs_ptr + offs_m, mask=mask_m, other=0.0)
        ws = tl.load(ws_ptr + n)
        y = acc * xs[:, None] * ws[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + n[None, :],
            y.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None],
        )
    else:
        tl.store(
            part_ptr + (pid_s * M + offs_m[:, None]) * N + n[None, :],
            acc,
            mask=mask_m[:, None],
        )


@triton.jit
def _fp8_gemv_reduce_kernel(
    part_ptr,
    xs_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    stride_om,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs // N
    n = offs % N
    mask = m < M
    acc = tl.load(part_ptr + offs, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLIT):
        acc += tl.load(part_ptr + s * M * N + offs, mask=mask, other=0.0)
    xs = tl.load(xs_ptr + m, mask=mask, other=0.0)
    ws = tl.load(ws_ptr + n, mask=mask, other=0.0)
    y = acc * xs * ws
    tl.store(out_ptr + m * stride_om + n, y.to(out_ptr.dtype.element_ty), mask=mask)


def _split(n_blocks: int, total_tiles: int, programs=None):
    split = min(total_tiles, max(1, round((_FP8_GEMV_PROGRAMS if programs is None else programs) / max(n_blocks, 1))))
    tiles_per_split = triton.cdiv(total_tiles, split)
    return triton.cdiv(total_tiles, tiles_per_split), tiles_per_split


_R6_SELECTIVE = envs.SGLANG_GFX942_GEMV_SELECTIVE.get()
_R6_LOGGED_SHAPES = set()


def fp8_gemv_rowwise(
    qinput: torch.Tensor,
    weight_nk: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    shuffled: bool,
) -> Optional[torch.Tensor]:
    """qinput fp8 [M, K] (M <= FP8_GEMV_MAX_M) x weight_nk fp8 [N, K] (shuffled: aiter shuffle_weight(layout=(16, 16)))
    with x_scale fp32 [M, 1] and w_scale fp32 [N, 1] -> out_dtype [M, N]; None when the shape is not served."""
    if qinput.dim() != 2:
        return None
    M, K = qinput.shape
    N = weight_nk.shape[0]
    selective = _R6_SELECTIVE and shuffled and (N, K) in ((1792, 5120), (1152, 5120)) and M <= 8
    if not selective and _FP8_GEMV_LAYOUT not in ("all", "shuffled" if shuffled else "rowmajor"):
        return None
    if not (
        0 < M <= min(FP8_GEMV_MAX_M, 16)
        and weight_nk.shape[1] == K
        and N % 16 == 0
        and K % 32 == 0
        and qinput.dtype == weight_nk.dtype
        and qinput.stride(1) == 1
        and weight_nk.is_contiguous()
        and x_scale.dtype == torch.float32
        and w_scale.dtype == torch.float32
        and x_scale.numel() == M
        and w_scale.numel() == N
    ):
        return None
    xs = x_scale.reshape(M).contiguous()
    ws = w_scale.reshape(N).contiguous()
    kb = max(1, min(8 if selective else _FP8_GEMV_KB, K // 32))
    total_tiles = triton.cdiv(K, kb * 32)
    n_blocks = N // 16
    split, tiles_per_split = _split(n_blocks, total_tiles, 1024 if selective else None)
    if selective and (N, K) not in _R6_LOGGED_SHAPES:
        import logging
        logging.getLogger(__name__).warning("R6_SELECTIVE_GEMV_EXECUTED M=%s N=%s K=%s", M, N, K)
        _R6_LOGGED_SHAPES.add((N, K))
    out = torch.empty((M, N), dtype=out_dtype, device=qinput.device)
    part = (
        torch.empty((split, M, N), dtype=torch.float32, device=qinput.device)
        if split > 1
        else out
    )
    _fp8_gemv_partial_kernel[(n_blocks, split)](
        qinput,
        weight_nk,
        xs,
        ws,
        out,
        part,
        M,
        N,
        K,
        qinput.stride(0),
        out.stride(0),
        SHUFFLED=shuffled,
        KB=kb,
        TILES_PER_SPLIT=tiles_per_split,
        SPLIT=split,
        num_warps=1 if selective else _FP8_GEMV_WARPS,
    )
    if split > 1:
        block = 1024
        _fp8_gemv_reduce_kernel[(triton.cdiv(M * N, block),)](
            part, xs, ws, out, M, N, out.stride(0), SPLIT=split, BLOCK=block, num_warps=4
        )
    return out


@triton.jit
def _bf16_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, 16)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros([16, BLOCK_N], dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[None, :].to(tl.int64) * stride_wn + (k0 + offs_k)[:, None],
            mask=mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(x, w)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def bf16_gemv(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype) -> Optional[torch.Tensor]:
    """x bf16 [M, K] (M <= LM_HEAD_GEMV_MAX_M) @ weight bf16 [N, K].T -> out_dtype [M, N]; None when not served."""
    if x.dim() != 2 or weight.dim() != 2:
        return None
    M, K = x.shape
    N = weight.shape[0]
    block_k = _LM_HEAD_BLOCK_K
    if not (
        0 < M <= min(LM_HEAD_GEMV_MAX_M, 16)
        and weight.shape[1] == K
        and K % block_k == 0
        and x.dtype == weight.dtype == torch.bfloat16
        and x.stride(1) == 1
        and weight.stride(1) == 1
    ):
        return None
    out = torch.empty((M, N), dtype=out_dtype, device=x.device)
    _bf16_gemv_kernel[(triton.cdiv(N, _LM_HEAD_BLOCK_N),)](
        x,
        weight,
        out,
        M,
        N,
        K,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        BLOCK_N=_LM_HEAD_BLOCK_N,
        BLOCK_K=block_k,
        NUM_STAGES=_LM_HEAD_STAGES,
        num_warps=_LM_HEAD_WARPS,
    )
    return out
