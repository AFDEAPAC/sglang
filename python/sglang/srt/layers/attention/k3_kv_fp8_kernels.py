"""Fused fp8 emission for the MLA prefill attention inputs (gfx942, e4m3fnuz, per-tensor scales).

kv_to_fp8(kv, k_pe, H, DN, DV):  kv = kv_b_proj output [L, H*(DN+DV)] bf16, k_pe [L, 1, DR] bf16
    -> k8 [L, H, DN+DR] fp8 (nope | broadcast rope), v8 [L, H, DV] fp8, sk [1], sv [1] (fp32 descales).
    Replaces torch.split + torch.cat + two dynamic per-tensor quants: one amax pass, one write pass.
quant(x) -> (x8, s): dynamic per-tensor quant of any tensor, two kernels.
Scale = amax / (fp8 max * 0.99), the margin matching the torch path it replaces.
"""
import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fnuz
FP8_MAX = torch.finfo(FP8).max
MARGIN = 0.99
BLOCK_L = 32


@triton.jit
def _kv_amax(kv_ptr, pe_ptr, amax_ptr, L, s_kv_l, s_pe_l, s_pe_d,
             H: tl.constexpr, DN: tl.constexpr, DV: tl.constexpr, DR: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    rm = rows < L
    dn = tl.arange(0, DN)
    dv = tl.arange(0, DV)
    dr = tl.arange(0, DR)
    pe = tl.load(pe_ptr + rows[:, None] * s_pe_l + dr[None, :] * s_pe_d, mask=rm[:, None], other=0.0)
    ak = tl.max(tl.max(tl.abs(pe.to(tl.float32)), axis=1), axis=0)
    av = 0.0
    for h in range(H):
        base = kv_ptr + rows[:, None] * s_kv_l + h * (DN + DV)
        x = tl.load(base + dn[None, :], mask=rm[:, None], other=0.0)
        ak = tl.maximum(ak, tl.max(tl.max(tl.abs(x.to(tl.float32)), axis=1), axis=0))
        y = tl.load(base + DN + dv[None, :], mask=rm[:, None], other=0.0)
        av = tl.maximum(av, tl.max(tl.max(tl.abs(y.to(tl.float32)), axis=1), axis=0))
    # |x| >= 0, so its int32 bit pattern orders like the float
    tl.atomic_max(amax_ptr, ak.to(tl.int32, bitcast=True))
    tl.atomic_max(amax_ptr + 1, av.to(tl.int32, bitcast=True))


@triton.jit
def _kv_quant(kv_ptr, pe_ptr, amax_ptr, k8_ptr, v8_ptr, sk_ptr, sv_ptr, L, s_kv_l, s_pe_l, s_pe_d,
              DENOM: tl.constexpr, H: tl.constexpr, DN: tl.constexpr, DV: tl.constexpr, DR: tl.constexpr,
              BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    sk = tl.maximum(tl.load(amax_ptr).to(tl.float32, bitcast=True) / DENOM, 1e-12)
    sv = tl.maximum(tl.load(amax_ptr + 1).to(tl.float32, bitcast=True) / DENOM, 1e-12)
    if pid == 0:
        tl.store(sk_ptr, sk)
        tl.store(sv_ptr, sv)
    ik = 1.0 / sk
    iv = 1.0 / sv
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    rm = rows < L
    dn = tl.arange(0, DN)
    dv = tl.arange(0, DV)
    dr = tl.arange(0, DR)
    pe = tl.load(pe_ptr + rows[:, None] * s_pe_l + dr[None, :] * s_pe_d, mask=rm[:, None], other=0.0)
    pe8 = (pe.to(tl.float32) * ik).to(tl.float8e4b8)
    for h in range(H):
        base = kv_ptr + rows[:, None] * s_kv_l + h * (DN + DV)
        x = tl.load(base + dn[None, :], mask=rm[:, None], other=0.0)
        kb = k8_ptr + rows[:, None] * (H * (DN + DR)) + h * (DN + DR)
        tl.store(kb + dn[None, :], (x.to(tl.float32) * ik).to(tl.float8e4b8), mask=rm[:, None])
        tl.store(kb + DN + dr[None, :], pe8, mask=rm[:, None])
        y = tl.load(base + DN + dv[None, :], mask=rm[:, None], other=0.0)
        vb = v8_ptr + rows[:, None] * (H * DV) + h * DV
        tl.store(vb + dv[None, :], (y.to(tl.float32) * iv).to(tl.float8e4b8), mask=rm[:, None])


def kv_to_fp8(kv, k_pe, H, DN, DV):
    L = kv.shape[0]
    kv = kv.reshape(L, H * (DN + DV))
    assert kv.stride(1) == 1
    pe = k_pe.reshape(L, -1)
    DR = pe.shape[1]
    amax = torch.zeros(2, dtype=torch.int32, device=kv.device)
    k8 = torch.empty((L, H, DN + DR), dtype=FP8, device=kv.device)
    v8 = torch.empty((L, H, DV), dtype=FP8, device=kv.device)
    sk = torch.empty(1, dtype=torch.float32, device=kv.device)
    sv = torch.empty(1, dtype=torch.float32, device=kv.device)
    grid = (triton.cdiv(L, BLOCK_L),)
    _kv_amax[grid](kv, pe, amax, L, kv.stride(0), pe.stride(0), pe.stride(1), H=H, DN=DN, DV=DV, DR=DR, BLOCK=BLOCK_L)
    _kv_quant[grid](kv, pe, amax, k8, v8, sk, sv, L, kv.stride(0), pe.stride(0), pe.stride(1),
                    DENOM=FP8_MAX * MARGIN, H=H, DN=DN, DV=DV, DR=DR, BLOCK=BLOCK_L)
    return k8, v8, sk, sv


@triton.jit
def _flat_amax(x_ptr, amax_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(x.to(tl.float32)), axis=0).to(tl.int32, bitcast=True))


@triton.jit
def _flat_quant(x_ptr, amax_ptr, y_ptr, s_ptr, N, DENOM: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    s = tl.maximum(tl.load(amax_ptr).to(tl.float32, bitcast=True) / DENOM, 1e-12)
    if pid == 0:
        tl.store(s_ptr, s)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(y_ptr + offs, (x.to(tl.float32) * (1.0 / s)).to(tl.float8e4b8), mask=m)


def quant(x):
    x = x.contiguous()
    n = x.numel()
    amax = torch.zeros(1, dtype=torch.int32, device=x.device)
    y = torch.empty(x.shape, dtype=FP8, device=x.device)
    s = torch.empty(1, dtype=torch.float32, device=x.device)
    blk = 8192
    grid = (triton.cdiv(n, blk),)
    _flat_amax[grid](x, amax, n, BLOCK=blk)
    _flat_quant[grid](x, amax, y, s, n, DENOM=FP8_MAX * MARGIN, BLOCK=blk)
    return y, s
